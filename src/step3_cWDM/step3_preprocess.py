import os
import re
import numpy as np
import nibabel as nib
import SimpleITK as sitk

# Optional imports kept for future tweaks (unused here)
from scipy.ndimage import zoom, affine_transform, median_filter, gaussian_filter, binary_erosion, binary_dilation  # noqa: F401
from skimage import filters  # noqa: F401

# ----------------------- CONFIGURE THESE -----------------------
# Set any of these to None or "" to skip that dataset.
NIFTI_DIR_TD_DATA   = None #"../../data/02_SEG/TD_data_INR_TotSeg"     # flat folder of .nii.gz; set None to skip
OUTPUT_DIR_TD_DATA  = "../../data/03_cWDM/TD_data_INR_TotSeg_Prepro"

# fastMRI expects subfolders: <folder>/<folder>.nii.gz
NIFTI_DIR_FASTMRI   = "../../data/02_SEG/fastMRI_INR_TotSeg"      # set None to skip
OUTPUT_DIR_FASTMRI  = "../../data/03_cWDM/fastMRI_INR_TotSeg_Prepro"

OFFSET_MM = 30  # [mm] offset around the patella for the mask
# ---------------------------------------------------------------


def safe_is_dir(p):
    return isinstance(p, str) and len(p) > 0 and os.path.isdir(p)


def main():
    print("Preprocess raw NIFTI files for inpainting")
    print(f"Using TD_data: {NIFTI_DIR_TD_DATA}")
    print(f"Output TD_data: {OUTPUT_DIR_TD_DATA}")
    print(f"Using fastMRI: {NIFTI_DIR_FASTMRI}")
    print(f"Output fastMRI: {OUTPUT_DIR_FASTMRI}")
    print(f"Offset: {OFFSET_MM} mm\n")

    # --------------------------- TD_data (NO SPLIT) ---------------------------
    td_enabled = safe_is_dir(NIFTI_DIR_TD_DATA) and isinstance(OUTPUT_DIR_TD_DATA, str) and len(OUTPUT_DIR_TD_DATA) > 0
    if td_enabled:
        print(f"TD_data input : {NIFTI_DIR_TD_DATA}")
        print(f"TD_data output: {OUTPUT_DIR_TD_DATA}")
        os.makedirs(OUTPUT_DIR_TD_DATA, exist_ok=True)
        nii_files_td = sorted([f for f in os.listdir(NIFTI_DIR_TD_DATA) if f.endswith(".nii.gz")])
        print(f"TD_data found {len(nii_files_td)} files")
        if len(nii_files_td) == 0:
            print("TD_data: No .nii.gz files found. Skipping TD_data.\n")
        else:
            process_td_nifti(OUTPUT_DIR_TD_DATA, nii_files_td, NIFTI_DIR_TD_DATA, OFFSET_MM)
    else:
        print("TD_data: Skipped (missing/invalid input/output path).\n")

    # --------------------------- fastMRI (80/20) ------------------------------
    fmri_enabled = safe_is_dir(NIFTI_DIR_FASTMRI) and isinstance(OUTPUT_DIR_FASTMRI, str) and len(OUTPUT_DIR_FASTMRI) > 0
    if fmri_enabled:
        print(f"fastMRI input : {NIFTI_DIR_FASTMRI}")
        print(f"fastMRI output: {OUTPUT_DIR_FASTMRI}")
        os.makedirs(OUTPUT_DIR_FASTMRI, exist_ok=True)

        # Scan patient folders for <folder>/<folder>.nii.gz
        nii_files_fastmri = []
        for patient_folder in sorted(os.listdir(NIFTI_DIR_FASTMRI)):
            patient_path = os.path.join(NIFTI_DIR_FASTMRI, patient_folder)
            if os.path.isdir(patient_path):
                combined_file = f"{patient_folder}.nii.gz"
                combined_path = os.path.join(patient_path, combined_file)
                if os.path.exists(combined_path):
                    # store relative path (patient_folder/combined_file)
                    nii_files_fastmri.append(os.path.join(patient_folder, combined_file))

        print(f"fastMRI found {len(nii_files_fastmri)} files")
        if len(nii_files_fastmri) == 0:
            print("fastMRI: No combined files (<folder>/<folder>.nii.gz) found. Skipping fastMRI.\n")
        else:
            process_fastmri_nifti(OUTPUT_DIR_FASTMRI, nii_files_fastmri, NIFTI_DIR_FASTMRI, OFFSET_MM)
    else:
        print("fastMRI: Skipped (missing/invalid input/output path).\n")

    if not td_enabled and not fmri_enabled:
        print("Nothing to do: provide at least one valid dataset input/output path in the constants at the top.")


# ===================== TD_data: test only (no split) =====================
def process_td_nifti(output_root, nii_files, raw_path, offset_mm):
    # Output dirs (no extra nested folder)
    test_root    = os.path.join(output_root, 'test')
    test_root_gt = os.path.join(output_root, 'test_GT')
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(test_root_gt, exist_ok=True)

    for file in nii_files:
        nifti_path = os.path.join(raw_path, file)
        print(f'[TD_data] Preprocessing NIFTI file {nifti_path}')

        # Expect name like subject_INR_TotSeg.nii.gz ; extract "subject"
        match = re.match(r'(.+?)_INR_', os.path.basename(file))
        if not match:
            print(f'[TD_data] Skipping file {file} - does not match regex pattern "(.+?)_INR_"')
            continue

        patient_folder_name = match.group(1)
        patient_folder_path = os.path.join(test_root, patient_folder_name)
        os.makedirs(patient_folder_path, exist_ok=True)

        base_filename = os.path.basename(file).split('.nii.gz')[0]

        # Output filenames
        segmentation_filename = f"{base_filename}_GT.nii.gz"
        diseased_filename     = f"{base_filename}_diseased.nii.gz"
        mask_filename         = f"{base_filename}_mask.nii.gz"

        # Run preprocessing
        result = preprocess_pipeline(nifti_path, output_root, offset_mm=offset_mm)
        if result is None:
            print(f"[TD_data] Skipping {patient_folder_name} - preprocess_pipeline returned None.")
            continue

        diseased_volume, mask_volume, segmented_image_numpy, new_affine = result

        print(f'[TD_data] Saving NIFTI files for {patient_folder_name}')

        # Save GT segmentation to test_3D_gt
        test_gt_patient_folder = os.path.join(test_root_gt, patient_folder_name)
        os.makedirs(test_gt_patient_folder, exist_ok=True)
        save_nifti(segmented_image_numpy, test_gt_patient_folder, segmentation_filename, new_affine)

        # Save diseased and mask volumes to test_3D
        save_nifti(diseased_volume, patient_folder_path, diseased_filename, new_affine)
        save_nifti(mask_volume,     patient_folder_path, mask_filename,     new_affine)


# ===================== fastMRI: 80/20 split =====================
def process_fastmri_nifti(output_root, nii_files, raw_path, offset_mm):
    # Deterministic split
    nii_files = sorted(nii_files)
    split_index = int(len(nii_files) * 0.8)
    train_files = set(nii_files[:split_index])
    test_files  = set(nii_files[split_index:])

    # Output dirs
    train_root   = os.path.join(output_root, 'train')
    test_root    = os.path.join(output_root, 'test')
    test_root_gt = os.path.join(output_root, 'test_GT')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(test_root_gt, exist_ok=True)

    for file in nii_files:
        nifti_path = os.path.join(raw_path, file)
        print(f'[fastMRI] Preprocessing NIFTI file {nifti_path}')

        # file may be like "folder/folder.nii.gz"
        leaf = os.path.basename(file)
        match = re.match(r'(.+?)_INR_', leaf)
        if not match:
            print(f'[fastMRI] Skipping file {file} - does not match regex pattern "(.+?)_INR_"')
            continue

        patient_folder_name = match.group(1)

        # Choose target split
        if file in train_files:
            target_root = train_root
        else:
            target_root = test_root

        patient_folder_path = os.path.join(target_root, patient_folder_name)
        os.makedirs(patient_folder_path, exist_ok=True)

        base_filename = leaf.split('.nii.gz')[0]

        # Output filenames
        segmentation_filename = f"{base_filename}_GT.nii.gz"
        diseased_filename     = f"{base_filename}_diseased.nii.gz"
        mask_filename         = f"{base_filename}_mask.nii.gz"

        # Run preprocessing
        result = preprocess_pipeline(nifti_path, output_root, offset_mm=offset_mm)
        if result is None:
            print(f"[fastMRI] Skipping {patient_folder_name} - preprocess_pipeline returned None.")
            continue

        diseased_volume, mask_volume, segmented_image_numpy, new_affine = result

        print(f'[fastMRI] Saving NIFTI files for {patient_folder_name}')

        # Save segmentation: train → inside train; test → into test_3D_gt as GT
        if file in train_files:
            save_nifti(segmented_image_numpy, patient_folder_path, segmentation_filename, new_affine)
        else:
            test_gt_patient_folder = os.path.join(test_root_gt, patient_folder_name)
            os.makedirs(test_gt_patient_folder, exist_ok=True)
            save_nifti(segmented_image_numpy, test_gt_patient_folder, segmentation_filename, new_affine)

        # Save diseased and mask volumes to the split folder
        save_nifti(diseased_volume, patient_folder_path, diseased_filename, new_affine)
        save_nifti(mask_volume,     patient_folder_path, mask_filename,     new_affine)


# ===================== Core pipeline utilities =====================
def preprocess_pipeline(input_path, save_path, offset_mm):
    print('[0] START')
    print(f'Processing image: {input_path}')
    img = nib.load(input_path)
    voxel_sizes = img.header.get_zooms()
    original_shape = img.shape
    original_affine = img.affine
    print(f'Original size: {original_shape}')
    print(f'Original Voxel sizes: {voxel_sizes}')

    data = img.get_fdata()

    print('--------Segmentation, find the Patella, define the mask--------')
    print("[1] Find the Patella outline based on segmentation, create a mask around the patella ...")

    centroid, binary_mask_patella_array = compute_patella_centroid(data)

    offset_surface = compute_anisotropic_offset_surface(binary_mask_patella_array, offset_mm, voxel_sizes)
    offset_mask_array = sitk.GetArrayFromImage(offset_surface)
    # ITK -> numpy orientation
    offset_mask_array = np.transpose(offset_mask_array, (2, 1, 0))

    cutoff_coord = centroid[2] + 5  # along Y after transpose used below
    mask = create_patella_offset_mask(np.transpose(binary_mask_patella_array, (2, 1, 0)),
                                      offset_mask_array, cutoff_coord)
    mask = np.transpose(mask, (2, 1, 0)).astype(bool)

    print("[2] Create the mask and diseased_image around the patella...")

    # Only keep mask where femur was originally present (exclude tibia=2 and fibula=4)
    not_tibia_or_fibula_mask = ((data != 2) & (data != 4))
    mask = mask & not_tibia_or_fibula_mask

    diseased_volume = np.copy(data)
    diseased_volume[mask] = 0  # Apply mask to volume

    mask_volume = np.zeros_like(data)
    mask_volume[mask] = 1

    # Normalize to [0, 1]
    diseased_volume = normalize_to_unit_range(diseased_volume)
    mask_volume     = normalize_to_unit_range(mask_volume)
    data            = normalize_to_unit_range(data)

    new_affine = original_affine
    return diseased_volume, mask_volume, data, new_affine


def normalize_to_unit_range(volume):
    volume = volume.astype(np.float32)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val > min_val:
        return (volume - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(volume)


def compute_patella_centroid(segmentation):
    # Labels assumption: patella=3
    patella_mask = (segmentation == 3).astype(np.uint8)

    voxel_indices = np.argwhere(patella_mask)
    if len(voxel_indices) == 0:
        print("Patella not found.")
        return (0, 0, 0), np.zeros_like(segmentation)

    centroid = np.mean(voxel_indices, axis=0).astype(int)
    x, y, z = centroid
    print(f"Patella found! Centroid (x,y,z): {centroid}, Voxel Count: {len(voxel_indices)}")
    # Return as (z, x, y) to match subsequent transpose usage
    return (z, x, y), patella_mask


def compute_anisotropic_offset_surface(binary_mask, offset_mm, voxel_spacing):
    # Convert offset from mm to voxel units per axis
    offset_voxels = [max(1, int(round(offset_mm / spacing))) for spacing in voxel_spacing]
    sitk_binary_mask = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    structuring_element = sitk.BinaryDilate(
        sitk_binary_mask,
        [offset_voxels[0], offset_voxels[1], offset_voxels[2]]  # radii in voxels for x,y,z
    )
    return structuring_element


def save_nifti(data, path, filename, affine):
    os.makedirs(path, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), os.path.join(path, filename))


def create_patella_offset_mask(binary_mask_patella_array, offset_mask_array, cutoff_coord):
    """
    Start with the dilated (offset) mask, remove the patella itself,
    and zero everything above cutoff_coord along Y.
    """
    final_mask = np.copy(offset_mask_array)

    # Remove the patella region from the mask
    final_mask[binary_mask_patella_array == True] = False  # noqa: E712

    # Zero everything above cutoff_coord along axis 1 (Y in transposed space)
    if cutoff_coord < final_mask.shape[1] - 1:
        final_mask[:, cutoff_coord + 1 :, :] = False

    return final_mask


if __name__ == "__main__":
    main()
