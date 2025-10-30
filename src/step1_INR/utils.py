from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss

from sklearn.preprocessing import MinMaxScaler

# (token, score)
_ORI_TOKENS = {
    "axial":     [("axial", 3), ("ax", 2), ("tra", 1), ("transverse", 1)],
    "coronal":   [("coronal", 3), ("cor", 2), ("frontal", 1)],
    "sagittal":  [("sagittal", 3), ("sag", 2), ("lateral", 1)],
}

def norm_grid(grid, xmin, xmax, smin=-1, smax=1):
    def min_max_scale(X, x_min, x_max, s_min, s_max):
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    return min_max_scale(X=grid, x_min=xmin, x_max=xmax, s_min=smin, s_max=smax)


def get_image_coordinate_grid_nib(image: nib.Nifti1Image):
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine
    (x, y, z) = image.shape

    label = []
    coordinates = []

    X = np.linspace(0, x - 1, x)
    Y = np.linspace(0, y - 1, y)
    Z = np.linspace(0, z - 1, z)


    points = np.meshgrid(X, Y, Z, indexing='ij')
    points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
    coordinates = list(nib.affines.apply_affine(img_affine, points))
    label = list(img_data.flatten())

    # convert to numpy array
    coordinates_array = np.array(coordinates, dtype=np.float32)
    label_array = np.array(label, dtype=np.float32)
    label_array_unnorm = label_array

    label_array = np.clip(label_array, np.quantile(label_array, 0.001), np.quantile(label_array, 0.999))
    label_array = (label_array - np.min(label_array)) / (np.max(label_array) - np.min(label_array))

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_array, s_min=-1, s_max=1)

    image_dict = {
        'affine': torch.tensor(img_affine),
        'origin': torch.tensor(np.array([0])),
        'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
        'dim': torch.tensor(np.array([x, y, z])),
        'intensity': torch.tensor(label_array_unnorm, dtype=torch.float32).view(-1, 1),
        'intensity_norm': torch.tensor(label_array, dtype=torch.float32).view(-1, 1),
        'coordinates': torch.tensor(coordinates_array, dtype=torch.float32),
        'coordinates_norm': torch.tensor(coordinates_arr_norm, dtype=torch.float32),
    }

    return image_dict

# ---- Orientation detection (token-based) ----
def _tokenize_filename(name: str) -> List[str]:
    # split on non-alphanumeric to avoid false matches (e.g., "message" != "sag")
    return re.findall(r"[A-Za-z0-9]+", name.lower())

def detect_orientation_from_name(filename: str) -> Tuple[Optional[str], int]:
    """
    Returns (orientation_key, score) where orientation_key ∈ {'axial','coronal','sagittal'}.
    If no orientation is recognized, returns (None, 0).
    """
    tokens = set(_tokenize_filename(Path(filename).name))
    best_ori, best_score = None, 0
    for ori, pairs in _ORI_TOKENS.items():
        for token, score in pairs:
            if token in tokens and score > best_score:
                best_ori, best_score = ori, score
    return best_ori, best_score

# ---- Dataset scanning ----
def scan_local_dataset_oriented(base_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Expects: base_dir/<patient_id>/<study_id>/*.nii[.gz]
    Returns:
        {
          patient_id: {
            study_id: {
              'axial': '/abs/path.nii.gz',
              'coronal': '/abs/path.nii.gz',
              'sagittal': '/abs/path.nii.gz'
            }
          }
        }
    Only includes orientations that can be inferred from filenames. If multiple
    candidates exist for an orientation, chooses the highest score, then shorter name.
    """
    patient_study_map: Dict[str, Dict[str, Dict[str, str]]] = {}
    base_dir = Path(base_dir)

    for patient_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        patient_id = patient_dir.name
        for study_dir in sorted([s for s in patient_dir.iterdir() if s.is_dir()]):
            study_id = study_dir.name
            nii_files = sorted(
                [*study_dir.glob("*.nii"), *study_dir.glob("*.nii.gz")],
                key=lambda p: p.name.lower()
            )
            if not nii_files:
                continue

            chosen: Dict[str, Tuple[str, int]] = {}  # ori -> (path, score)
            for f in nii_files:
                ori, score = detect_orientation_from_name(f.name)
                if not ori or score <= 0:
                    continue
                if ori not in chosen:
                    chosen[ori] = (str(f.resolve()), score)
                else:
                    prev_path, prev_score = chosen[ori]
                    if score > prev_score or (score == prev_score and len(f.name) < len(Path(prev_path).name)):
                        chosen[ori] = (str(f.resolve()), score)

            if chosen:
                patient_study_map.setdefault(patient_id, {})
                patient_study_map[patient_id][study_id] = {k: v[0] for k, v in chosen.items()}

    return patient_study_map

def build_patient_study_list(
    patient_study_map: Dict[str, Dict[str, Dict[str, str]]],
    subset_start: int = 0,
    subset_end: Optional[int] = None,
) -> List[Tuple[str, Set[str]]]:
    """
    Converts nested dict to a list of (patient_id, {study_ids}), optionally sliced.
    """
    items = [(pid, set(studies.keys())) for pid, studies in patient_study_map.items()]
    return items[subset_start:] if subset_end is None else items[subset_start:subset_end]