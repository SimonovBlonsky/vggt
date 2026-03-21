from __future__ import annotations

import cv2
import numpy as np


def get_rot_err(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Match reloc3r.utils.metric.get_rot_err exactly."""
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1, 3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.0
    return float(rot_err[0])


def get_transl_ang_err(dir_a: np.ndarray, dir_b: np.ndarray) -> float:
    """Match reloc3r.utils.metric.get_transl_ang_err exactly."""
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)
    return float(err)


def error_auc(r_error: np.ndarray, t_errors: np.ndarray, thresholds=(5, 10, 20)) -> dict[str, float]:
    """Match reloc3r.utils.metric.error_auc exactly."""
    error_matrix = np.concatenate((r_error[:, None], t_errors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        if last_index == 0:
            aucs.append(0.0)
            continue
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(float(np.trapz(y, x) / thr))

    return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}
