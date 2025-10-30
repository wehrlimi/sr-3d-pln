#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

in_dir="../../data/01_INR/fastMRI_INR"
#in_dir="../../data/01_INR/TD_data_INR"
out_root="../../data/02_SEG/fastMRI_INR_TotSeg"
#out_root="../../data/02_SEG/TD_data_INR_TotSeg"

# Collect input files
mapfile -d '' files < <(find "$in_dir" -type f -name '*.nii.gz' -print0 | sort -z)
if (( ${#files[@]} == 0 )); then
  echo "No .nii.gz files found under: $in_dir"
  exit 1
fi

mkdir -p "$out_root"

for f in "${files[@]}"; do
  fname="$(basename "$f" .nii.gz)"
  outdir="$out_root/${fname}_TotSeg"
  mkdir -p "$outdir"

  echo "=== Processing $fname ==="

  # Run both segmentation tasks into the same folder
  echo "Running TotalSegmentator --task total_mr ..."
  TotalSegmentator -i "$f" -o "$outdir" --task total_mr

  echo "Running TotalSegmentator --task appendicular_bones_mr ..."
  TotalSegmentator -i "$f" -o "$outdir" --task appendicular_bones_mr

  # Keep only selected structures
  echo "Cleaning up unnecessary files..."
  find "$outdir" -type f -name '*.nii.gz' \
    ! -iname '*femur*' \
    ! -iname '*fibula*' \
    ! -iname '*tibia*' \
    ! -iname '*patella*' \
    -delete
  echo "=== Done with $fname ==="
done