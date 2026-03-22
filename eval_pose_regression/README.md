# VGGT Relative Pose Evaluation

This folder provides a quick relative-pose benchmark for VGGT on the same pair lists used by `reloc3r`:

- `reloc3r/data/scannet1500`
- `reloc3r/data/megadepth1500`

## What is measured

The evaluator matches `reloc3r/eval_relpose.py` exactly for metrics:

- rotation angular error
- translation-direction angular error
- `auc@5`, `auc@10`, `auc@20`, computed from `max(R error, T error)`

## How pose is predicted

For each image pair `(I1, I2)`:

1. load the pair through the shared reloc3r-style dataset crop in `/home/chenguyuan/code/NeurIPS26/datasets`
2. apply VGGT's model-specific resize or pad stage on top of that shared crop
3. run only VGGT's pose path: `aggregator -> camera_head`
4. convert `pose_enc` to camera extrinsics with `pose_encoding_to_extri_intri`
5. form the relative pose as `T_2to1 = E1 @ inv(E2)`

Here `E1` and `E2` are OpenCV-style `camera-from-world` extrinsics predicted by VGGT.

## Files

- `eval_relpose.py`: main entry point for both datasets
- `/home/chenguyuan/code/NeurIPS26/datasets/scannet1500.py`: shared ScanNet1500 reader
- `/home/chenguyuan/code/NeurIPS26/datasets/megadepth_valid.py`: shared MegaDepth1500 reader
- `/home/chenguyuan/code/NeurIPS26/eval/utils/metric.py`: shared reloc3r-compatible metrics
- `eval_scannet1500.sh`: convenience wrapper for ScanNet1500
- `eval_megadepth1500.sh`: convenience wrapper for MegaDepth1500

## Example usage

```bash
cd /home/chenguyuan/code/NeurIPS26/vggt/eval_pose_regression
bash eval_scannet1500.sh
bash eval_megadepth1500.sh
```

You can override the checkpoint or preprocessing mode, for example:

```bash
bash eval_scannet1500.sh --model-path facebook/VGGT-1B --preprocess-mode pad
```

## Speed output

The script also prints per-pair timing statistics:

- `ms_per_pair_model_only`
- `ms_per_pair_end_to_end`
- `pairs_per_sec_model_only`
- `pairs_per_sec_end_to_end`

This makes it straightforward to compare VGGT against `reloc3r` and the existing DA3 evaluator.
