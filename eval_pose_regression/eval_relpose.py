from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
VGGT_ROOT = REPO_ROOT / "vggt"
for path in (REPO_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import datasets.megadepth_valid as megadepth_valid_module  # noqa: E402
import datasets.scannet1500 as scannet1500_module  # noqa: E402
from utils.metric import error_auc, get_rot_err, get_transl_ang_err  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


TO_TENSOR = tvf.ToTensor()


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick relative-pose evaluation for VGGT")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scannet1500", "megadepth1500"],
        help="Evaluation dataset name",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="facebook/VGGT-1B",
        help="Hugging Face repo id or local pretrained checkpoint directory",
    )
    parser.add_argument(
        "--scannet-root",
        type=str,
        default=str(REPO_ROOT / "reloc3r" / "data" / "scannet1500"),
        help="Path to reloc3r/data/scannet1500",
    )
    parser.add_argument(
        "--megadepth-root",
        type=str,
        default=str(REPO_ROOT / "reloc3r" / "data" / "megadepth1500"),
        help="Path to reloc3r/data/megadepth1500",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of pairs per forward pass")
    parser.add_argument(
        "--dataset-width",
        type=int,
        default=512,
        help="Shared reloc3r-style dataset crop width",
    )
    parser.add_argument(
        "--dataset-height",
        type=int,
        default=384,
        help="Shared reloc3r-style dataset crop height",
    )
    parser.add_argument("--seed", type=int, default=777, help="Dataset RNG seed")
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="VGGT preprocessing mode applied after the shared dataset crop",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=518,
        help="Target VGGT width or square size after the shared dataset crop",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Autocast dtype for the aggregator. Camera head runs after the autocast block.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=-1,
        help="Debug option: only evaluate the first N pairs (-1 means all)",
    )
    parser.add_argument(
        "--save-errors",
        type=str,
        default="",
        help="Optional path to save per-pair rotation/translation errors as an npz file",
    )
    return parser


def batched_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield range(start, min(total, start + batch_size))


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_autocast_dtype(device: torch.device, dtype_name: str) -> torch.dtype | None:
    if device.type != "cuda" or dtype_name == "float32":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    capability_major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if capability_major >= 8 else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def as_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (..., 3, 4) or (..., 4, 4) extrinsics, got {tuple(extrinsics.shape)}")
    out = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    out = out.view(*([1] * (extrinsics.ndim - 2)), 4, 4).repeat(*extrinsics.shape[:-2], 1, 1)
    out[..., :3, :4] = extrinsics
    return out


def build_dataset(args: argparse.Namespace):
    common_kwargs = dict(
        resolution=(args.dataset_width, args.dataset_height),
        seed=args.seed,
        transform=TO_TENSOR,
    )
    if args.dataset == "scannet1500":
        scannet1500_module.DATA_ROOT = args.scannet_root
        return scannet1500_module.ScanNet1500(**common_kwargs)
    if args.dataset == "megadepth1500":
        megadepth_valid_module.DATA_ROOT = args.megadepth_root
        return megadepth_valid_module.MegaDepth_valid(**common_kwargs)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def pair_id_from_views(views: list[dict]) -> str:
    left = f"{views[0]['label']}/{views[0]['instance']}"
    right = f"{views[1]['label']}/{views[1]['instance']}"
    return f"{left} || {right}"


def gt_pose2to1_from_views(views: list[dict]) -> np.ndarray:
    return np.linalg.inv(views[0]["camera_pose"]) @ views[1]["camera_pose"]


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def preprocess_views_for_vggt(pair_views_batch: list[list[dict]], mode: str, target_size: int) -> torch.Tensor:
    flat_views = [view for views in pair_views_batch for view in views]
    imgs = torch.stack([view["img"].float() for view in flat_views], dim=0)
    _, _, height, width = imgs.shape

    if mode == "crop":
        new_width = target_size
        new_height = round_to_multiple(height * (new_width / width), 14)
        imgs = F.interpolate(imgs, size=(new_height, new_width), mode="bilinear", align_corners=False)
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            imgs = imgs[:, :, start_y : start_y + target_size, :]
    elif mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round_to_multiple(height * (new_width / width), 14)
        else:
            new_height = target_size
            new_width = round_to_multiple(width * (new_height / height), 14)
        imgs = F.interpolate(imgs, size=(new_height, new_width), mode="bilinear", align_corners=False)

        h_padding = target_size - imgs.shape[-2]
        w_padding = target_size - imgs.shape[-1]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            imgs = F.pad(imgs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
    else:
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    return imgs.view(len(pair_views_batch), 2, *imgs.shape[1:])


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    infer_dtype = resolve_autocast_dtype(device, args.dtype)
    dataset = build_dataset(args)
    total_pairs = len(dataset) if args.max_pairs < 0 else min(len(dataset), args.max_pairs)

    print(f"Loading VGGT model from: {args.model_path}")
    model = VGGT.from_pretrained(args.model_path).to(device)
    model.eval()

    rerrs: list[float] = []
    terrs: list[float] = []
    pair_ids: list[str] = []
    processed_pairs = 0

    preprocess_time = 0.0
    model_time = 0.0
    metric_time = 0.0

    total_start = time.perf_counter()
    pbar = tqdm(total=total_pairs, desc=f"Evaluating {args.dataset}", unit="pair")
    try:
        for batch_ids in batched_indices(total_pairs, args.batch_size):
            pair_views_batch = [dataset[i] for i in batch_ids]

            t0 = time.perf_counter()
            imgs_cpu = preprocess_views_for_vggt(pair_views_batch, args.preprocess_mode, args.target_size)
            imgs = imgs_cpu.to(device, non_blocking=True)
            preprocess_time += time.perf_counter() - t0

            synchronize_if_cuda(device)
            t1 = time.perf_counter()
            with autocast_context(device, infer_dtype):
                aggregated_tokens_list, _ = model.aggregator(imgs)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1].float()
            pred_ext, _ = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
            synchronize_if_cuda(device)
            model_time += time.perf_counter() - t1

            t2 = time.perf_counter()
            pred_ext = as_homogeneous(pred_ext)
            pred_rel_2to1 = pred_ext[:, 0] @ torch.linalg.inv(pred_ext[:, 1])
            pred_rel_2to1 = pred_rel_2to1.detach().cpu().numpy()

            for sid, views in enumerate(pair_views_batch):
                gt_pose2to1 = gt_pose2to1_from_views(views)
                pr_pose2to1 = pred_rel_2to1[sid]

                rerr = get_rot_err(pr_pose2to1[:3, :3], gt_pose2to1[:3, :3])

                transl = pr_pose2to1[:3, 3]
                gt_transl = gt_pose2to1[:3, 3]
                terr = get_transl_ang_err(transl, gt_transl)

                rerrs.append(rerr)
                terrs.append(terr)
                pair_ids.append(pair_id_from_views(views))
            metric_time += time.perf_counter() - t2

            processed_pairs += len(pair_views_batch)
            elapsed = time.perf_counter() - total_start
            pbar.update(len(pair_views_batch))
            pbar.set_postfix(pair_per_s=f"{processed_pairs / max(elapsed, 1e-6):.2f}")
    finally:
        pbar.close()

    total_time = time.perf_counter() - total_start
    rerrs_np = np.asarray(rerrs, dtype=np.float32)
    terrs_np = np.asarray(terrs, dtype=np.float32)
    aucs = error_auc(rerrs_np, terrs_np, thresholds=[5, 10, 20])

    speed = {
        "pairs": int(len(rerrs_np)),
        "batch_size": int(args.batch_size),
        "preprocess_time_s": preprocess_time,
        "model_time_s": model_time,
        "metric_time_s": metric_time,
        "total_time_s": total_time,
        "ms_per_pair_model_only": 1000.0 * model_time / max(len(rerrs_np), 1),
        "ms_per_pair_end_to_end": 1000.0 * total_time / max(len(rerrs_np), 1),
        "pairs_per_sec_model_only": len(rerrs_np) / max(model_time, 1e-6),
        "pairs_per_sec_end_to_end": len(rerrs_np) / max(total_time, 1e-6),
    }

    print(f"In total {len(rerrs_np)} pairs")
    print(json.dumps(aucs, indent=2))
    print(json.dumps(speed, indent=2))

    if args.save_errors:
        save_path = Path(args.save_errors)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, pair_id=np.asarray(pair_ids), rerr=rerrs_np, terr=terrs_np)
        print(f"Saved per-pair errors to {save_path}")

    return aucs


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    evaluate(args)
