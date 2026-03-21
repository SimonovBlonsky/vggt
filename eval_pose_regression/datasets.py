from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairSample:
    image1: str
    image2: str
    pose1_c2w: np.ndarray
    pose2_c2w: np.ndarray
    dataset: str
    pair_id: str

    @property
    def gt_pose2to1(self) -> np.ndarray:
        """Relative pose from camera-2 coordinates to camera-1 coordinates."""
        return np.linalg.inv(self.pose1_c2w) @ self.pose2_c2w


class ScanNet1500Pairs:
    """
    Lightweight reader for reloc3r/data/scannet1500.

    Notes:
    - Pair definitions come from test.npz['name'].
    - GT pose files are loaded directly from the extracted ScanNet1500 folder.
    - Pose files are treated as camera-to-world transforms, matching reloc3r's eval logic.
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.pairs_path = self.data_root / "test.npz"
        self.scenes_root = self.data_root / "scannet_test_1500"
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Cannot find pair file: {self.pairs_path}")
        if not self.scenes_root.exists():
            raise FileNotFoundError(
                f"Cannot find extracted ScanNet1500 folder: {self.scenes_root}. "
                "Expected reloc3r/data/scannet1500/scannet_test_1500/"
            )

        with np.load(self.pairs_path) as data:
            self.pair_names = data["name"]

    def __len__(self) -> int:
        return len(self.pair_names)

    def __getitem__(self, idx: int) -> PairSample:
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]
        scene_dir = self.scenes_root / f"scene{int(scene_name):04d}_{int(scene_sub_name):02d}"

        image1 = scene_dir / "color" / f"{int(name1)}.jpg"
        image2 = scene_dir / "color" / f"{int(name2)}.jpg"
        pose1 = scene_dir / "pose" / f"{int(name1)}.txt"
        pose2 = scene_dir / "pose" / f"{int(name2)}.txt"

        if not image1.exists() or not image2.exists():
            raise FileNotFoundError(f"Missing image pair at index {idx}: {image1}, {image2}")
        if not pose1.exists() or not pose2.exists():
            raise FileNotFoundError(f"Missing pose pair at index {idx}: {pose1}, {pose2}")

        pair_id = f"scene{int(scene_name):04d}_{int(scene_sub_name):02d}/{int(name1)}-{int(name2)}"
        return PairSample(
            image1=str(image1),
            image2=str(image2),
            pose1_c2w=np.loadtxt(pose1).astype(np.float32),
            pose2_c2w=np.loadtxt(pose2).astype(np.float32),
            dataset="scannet1500",
            pair_id=pair_id,
        )


class MegaDepth1500Pairs:
    """
    Lightweight reader for reloc3r/data/megadepth1500.

    Notes:
    - Pair definitions come from megadepth_test_pairs.txt.
    - GT camera poses follow reloc3r/reloc3r/datasets/megadepth_valid.py exactly:
      camera_pose = inv(metadata[view_idx]['pose']).
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.meta_path = self.data_root / "megadepth_meta_test.npz"
        self.pairs_path = self.data_root / "megadepth_test_pairs.txt"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Cannot find metadata file: {self.meta_path}")
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Cannot find pair list: {self.pairs_path}")

        self.metadata = np.load(self.meta_path, allow_pickle=True)
        self.pairs = []
        with self.pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image1, image2 = line.split()
                self.pairs.append((image1, image2))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairSample:
        image1_rel, image2_rel = self.pairs[idx]
        image1 = self.data_root / image1_rel
        image2 = self.data_root / image2_rel
        if not image1.exists() or not image2.exists():
            raise FileNotFoundError(f"Missing image pair at index {idx}: {image1}, {image2}")

        meta1 = self.metadata[image1_rel].item()
        meta2 = self.metadata[image2_rel].item()
        pose1_c2w = np.linalg.inv(np.asarray(meta1["pose"], dtype=np.float32)).astype(np.float32)
        pose2_c2w = np.linalg.inv(np.asarray(meta2["pose"], dtype=np.float32)).astype(np.float32)

        return PairSample(
            image1=str(image1),
            image2=str(image2),
            pose1_c2w=pose1_c2w,
            pose2_c2w=pose2_c2w,
            dataset="megadepth1500",
            pair_id=f"{image1_rel} {image2_rel}",
        )
