#!/usr/bin/env python3
"""
Fuse bone masks into a single label map per patient folder.

- Femur: choose the side (left/right) with MORE nonzero voxels.
- Tibia, Fibula, Patella: include all sides present; both sides share the same label.
- Labels: femur=1, tibia=2, patella=3, fibula=4
- Output: <foldername>.nii.gz saved inside the folder.
"""

# ===============================================================
# ===================== CONFIGURATION ===========================
# ===============================================================

# Path to directory that contains patient folders such as .../patientA_TotSeg/
ROOT_DIR = "../../data/02_SEG/fastMRI_INR_TotSeg"

# Glob filter for selecting patient folders; usually "*"
FOLDER_GLOB = "*"

# If True → only print actions, don’t write files
DRY_RUN = False

# Less console output
QUIET = False

# ===============================================================
# ======================= SCRIPT CODE ===========================
# ===============================================================

import sys
from pathlib import Path
import re
import numpy as np
import nibabel as nib


def load_mask(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    mask = (data > 0).astype(np.uint8)
    return img, mask


def find_one_or_bilateral(folder: Path, keyword: str):
    kw = keyword.lower()
    candidates = list(folder.glob("*.nii.gz"))

    left = right = any_one = None
    for p in candidates:
        name = p.name.lower()
        if kw not in name:
            continue

        if ("left" in name or "_l" in name or re.search(r"\bL\b", name)) and left is None:
            left = p
        elif ("right" in name or "_r" in name or re.search(r"\bR\b", name)) and right is None:
            right = p
        elif any_one is None:
            any_one = p

    return {"left": left, "right": right, "any": any_one}


def pick_femur_path(folder: Path):
    fem = find_one_or_bilateral(folder, "femur")

    if fem["left"] or fem["right"]:
        best_path = None
        best_count = -1
        for side in ("left", "right"):
            p = fem[side]
            if p and p.exists():
                try:
                    _, m = load_mask(p)
                    cnt = int(np.count_nonzero(m))
                    if cnt > best_count:
                        best_count = cnt
                        best_path = p
                except Exception:
                    pass
        return best_path

    return fem["any"]


def union_paths(folder: Path, keyword: str):
    found = find_one_or_bilateral(folder, keyword)
    paths = []
    if found["left"]:
        paths.append(found["left"])
    if found["right"]:
        paths.append(found["right"])
    if not paths and found["any"]:
        paths.append(found["any"])
    return paths


def get_output_name(folder: Path):
    return folder / f"{folder.name}.nii.gz"


def fuse_patient_folder(folder: Path, dry_run=False, verbose=True):
    if verbose:
        print(f"== Processing {folder.name}")

    femur_path = pick_femur_path(folder)
    tibia_paths = union_paths(folder, "tibia")
    patella_paths = union_paths(folder, "patella")
    fibula_paths = union_paths(folder, "fibula")

    if not any([femur_path, tibia_paths, patella_paths, fibula_paths]):
        if verbose:
            print("   No masks found — skipping")
        return False

    # reference image
    ref_path = femur_path or (tibia_paths + patella_paths + fibula_paths)[0]
    ref_img = nib.load(str(ref_path))
    shape = ref_img.shape
    affine = ref_img.affine
    header = ref_img.header

    out = np.zeros(shape, dtype=np.uint8)

    def add_label(paths, label, name):
        total = 0
        for p in paths:
            img, mask = load_mask(p)
            if img.shape != shape:
                raise ValueError(f"Shape mismatch: {p}")
            write = (mask > 0) & (out == 0)
            out[write] = label
            n = int(np.count_nonzero(write))
            total += n
            if verbose:
                print(f"   + {name:<8} {p.name:<30} -> {n} vox")
        return total

    vox = 0

    # Femur = 1
    if femur_path:
        _, fmask = load_mask(femur_path)
        write = (fmask > 0) & (out == 0)
        out[write] = 1
        n = int(np.count_nonzero(write))
        vox += n
        if verbose:
            print(f"   + femur    {femur_path.name:<30} -> {n} vox")

    # Tibia = 2
    vox += add_label(tibia_paths, 2, "tibia")

    # Patella = 3
    vox += add_label(patella_paths, 3, "patella")

    # Fibula = 4
    vox += add_label(fibula_paths, 4, "fibula")

    if vox == 0:
        if verbose:
            print("   No voxels written — skip")
        return False

    out_img = nib.Nifti1Image(out, affine, header)
    out_img.set_data_dtype(np.uint8)

    out_path = get_output_name(folder)
    if dry_run:
        if verbose:
            print(f"   [DRY] Would save to {out_path}")
        return True

    nib.save(out_img, str(out_path))
    if verbose:
        print(f"   Saved: {out_path}")

    return True


def main():
    root = Path(ROOT_DIR).resolve()
    print("FUSING")
    if not root.is_dir():
        print(f"ERROR: folder not found: {root}")
        sys.exit(1)

    folders = [p for p in root.glob(FOLDER_GLOB) if p.is_dir()]
    if not folders:
        print("No folders found")
        sys.exit(1)

    n_ok = n_total = 0

    for folder in sorted(folders):
        if not any(folder.glob("*.nii.gz")):
            continue
        n_total += 1
        try:
            ok = fuse_patient_folder(folder, dry_run=DRY_RUN, verbose=not QUIET)
            if ok:
                n_ok += 1
        except Exception as e:
            print(f"!! Error on {folder.name}: {e}")

    print(f"\nDone. Saved {n_ok} / {n_total} folders.")


if __name__ == "__main__":
    main()
