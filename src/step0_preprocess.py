import os
import pdb
import pydicom
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
import faulthandler

faulthandler.enable()

from dicom2nifti import convert_directory  # imported but we call it in a child process

# Define global paths (modify as needed)
ROOT_FOLDER_FASTMRI = r'../data/knee_mri_clinical_seq_batch2'
ROOT_FOLDER_TD_data = ''  # Raw DICOMS from patients with TD
OUTPUT_PATH = r'../data/00_sorted_nifti_for_INR'  # Where your nifti's will be stored
EXCEL_NAME = 'INR_Excel_Analysis.xlsx'

# Define output folders
OUTPUT_EXCEL_PATH = os.path.join(OUTPUT_PATH, EXCEL_NAME)
FASTMRI_OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, 'fastMRI_nifti')
TD_OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, 'TD_data_nifti')

# Define filtering criteria (ensure these are set properly)
ACCEPTED_DESCRIPTIONS = {
    'Axial': [
        'pd_fs_tra', 'pd_tse_fs_tra', 'pd_tse_fs_drb_tra', 'pd_tse_fs-dixon_tra_W',
        'tra_pd', 'tra_pd_fs', 'ax pd', 'pd axial fat sat', 'pd axial f/s',
        'pd axial fat sat repeat', 'pd axial fat sat rpt', 'pd axial fat sat_fil_1',
        'pd axial fs repeat', 'pd axial fs rpt'
    ],
    'Coronal': ['cor', 'coronal', 'frontal'],
    'Sagittal': ['sag', 'sagittal', 'lateral']
}

def fastmri_importer(root_folder_fastMRI):
    """
    Imports all metadata from the fastMRI folder. We rely on the folder structure, as not all metadata
    (such as study number) are present in the .dcm files. However, the series description (PD_fs_ax)
    is based on the .dcm files and added to the metadata info.
    """
    metadata_list = []
    if root_folder_fastMRI == '':
        return metadata_list
    for subdir, _, files in os.walk(root_folder_fastMRI):
        dicom_files = sorted([file for file in files if file.lower().endswith('.dcm')])
        if dicom_files:
            dicom_file_path = os.path.join(subdir, dicom_files[0])
            description = extract_description_dcm(dicom_file_path)
            patient_folder, study_folder, MR_folder = subdir.split(os.sep)[-3:]
            volume_id = f"{patient_folder}_{study_folder}_{MR_folder}_{description}"
            metadata = {
                'DICOM Path': os.path.dirname(dicom_file_path),
                'Patient Folder': patient_folder,
                'Personal ID': None,
                'Patient ID': patient_folder,  # Take the patient folder as Patient ID in fastMRI
                'Request ID': None,
                'Study ID': study_folder,
                'Series ID': None,  # Not available in fastMRI
                'MR ID': MR_folder,
                'Description': description,
                'Volume ID': volume_id,
                'Nifti Path': None
            }
            metadata_list.append(metadata)
    return metadata_list

def ukbb_importer(root_folder_UKBB):
    """
    Imports all metadata from the UKBB folder using the CONTENT.xml files.
    """
    metadata_list = []
    if root_folder_UKBB == '':
        return metadata_list
    for patient_folder in os.listdir(root_folder_UKBB):
        patient_path = os.path.join(root_folder_UKBB, patient_folder)
        xml_file = os.path.join(patient_path, 'SECTRA', 'CONTENT.xml')
        if os.path.exists(xml_file):
            descriptions = extract_description_xml(xml_file)
            for desc in descriptions:
                volume_id = f"{desc['Patient ID']}_{desc['Personal ID']}_{desc['Request ID']}_{desc['Study ID']}_{desc['Series ID']}_{desc['Description']}"
                desc['Volume ID'] = volume_id
                metadata = {
                    'DICOM Path': os.path.join(patient_path, 'DICOM', desc['Patient ID'], desc['Request ID'],
                                               desc['Study ID'], desc['Series ID']),
                    'Patient Folder': os.path.basename(patient_path),
                    'Personal ID': desc['Personal ID'],
                    'Patient ID': desc['Patient ID'],
                    'Request ID': desc['Request ID'],
                    'Study ID': desc['Study ID'],
                    'Series ID': desc['Series ID'],
                    'MR ID': None,  # Not available in UKBB
                    'Description': desc['Description'],
                    'Volume ID': volume_id,
                    'Nifti Path': None
                }
                metadata_list.append(metadata)
    return metadata_list

def extract_description_dcm(dicom_file):
    """Extract SeriesDescription from a DICOM file, safely (no pixel data)."""
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        return ds.get('SeriesDescription', 'N/A')
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return 'N/A'

def extract_description_xml(xml_path):
    """Extracts series descriptions from a CONTENT.xml file."""
    description_list = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for patient in root.findall('.//patient'):
        patient_id = patient.get('id', 'UnknownPatientID')
        personal_id = patient.find('.//personal_id')
        personal_id = personal_id.text if personal_id is not None else 'UnknownPersonalID'
        for request in root.findall('.//request'):
            request_id = request.get('id', 'UnknownRequestID')
            for study in request.findall('.//study'):
                study_id = study.get('id', 'UnknownStudyID')
                for series in study.findall('.//series'):
                    series_id = series.get('id', 'UnknownSeriesID')
                    desc_node = series.find('.//description')
                    description = desc_node.text if desc_node is not None else 'NoDescription'
                    description_list.append({
                        'Patient ID': patient_id,
                        'Personal ID': personal_id,
                        'Request ID': request_id,
                        'Study ID': study_id,
                        'Series ID': series_id,
                        'Description': description
                    })
    return description_list

def select_volumes(df, accepted_descriptions):
    """
    Marks volumes selected for conversion (Axial, Coronal, Sagittal) per Study ID,
    while preserving the original data in the dataframe. Adds a 'selected_for_conversion' column.
    """
    if df is None or df.empty:
        return df

    result_df = df.copy()
    result_df['selected_for_conversion'] = False

    axial_keywords = accepted_descriptions.get('Axial', [])

    for study_id, group in result_df.groupby('Study ID'):
        selected_views = {}

        pd_axial_match = group[
            group["Description"].str.lower().str.contains('|'.join(axial_keywords), na=False)
        ]
        if pd_axial_match.empty:
            continue

        for view, keywords in accepted_descriptions.items():
            matching_rows = group[group["Description"].str.lower().str.contains('|'.join(keywords), na=False)]
            if not matching_rows.empty:
                pd_filtered = matching_rows[matching_rows['Description'].str.contains("pd", case=False, na=False)]
                if not pd_filtered.empty:
                    selected_views[view] = pd_filtered.sort_values("Description").iloc[0]
                else:
                    selected_views[view] = matching_rows.sort_values("Description").iloc[0]

        if len(selected_views) == 3:
            selected_indices = [
                selected_views["Axial"].name,
                selected_views["Coronal"].name,
                selected_views["Sagittal"].name
            ]
            result_df.loc[selected_indices, 'selected_for_conversion'] = True

    return result_df

def process_selected_dicom_folders(selected_folders, main_output_folder, sheet_name, original_df):
    """
    Processes each selected DICOM folder by converting to NIfTI in a temp folder, renaming,
    and storing in a structured patient/study folder. Also logs conversion errors with reasons.
    """
    processed_patient_ids = set()
    nifti_updates = []

    for folder_info in selected_folders:
        dicom_folder = folder_info["DICOM Path"]
        patient_id = folder_info["Patient ID"]
        volume_id = folder_info["Volume ID"]
        study_name = folder_info["Study ID"]

        if patient_id not in processed_patient_ids:
            processed_patient_ids.add(patient_id)

        # Unique temp dir per series to avoid cross-talk
        temp_folder = os.path.join(main_output_folder, f"temp_conversion_{uuid.uuid4().hex}")

        error_message, nifti_path = convert_dicom_to_nifti_temp(dicom_folder, temp_folder), None
        if error_message:
            folder_info["Nifti Path"] = "not converted"
            print(error_message)
        else:
            nifti_path = rename_and_move_nifti_files(temp_folder, main_output_folder, volume_id, patient_id, study_name)
            folder_info["Nifti Path"] = nifti_path if nifti_path else "not converted"

        nifti_updates.append({
            'Volume ID': volume_id,
            'Nifti Path': folder_info["Nifti Path"]
        })

    for update in nifti_updates:
        mask = original_df['Volume ID'] == update['Volume ID']
        original_df.loc[mask, 'Nifti Path'] = update['Nifti Path']

    return original_df

def convert_dicom_to_nifti_temp(dicom_folder, temp_folder):
    """
    Converts a folder of DICOM files to NIfTI format in a temporary folder.
    Runs dicom2nifti in a child Python process to isolate any native segfaults.
    Returns: None if OK, else error message string.
    """
    try:
        os.makedirs(temp_folder, exist_ok=True)

        helper = textwrap.dedent("""
        import sys, os
        import faulthandler; faulthandler.enable()
        from dicom2nifti import convert_directory
        os.makedirs(sys.argv[2], exist_ok=True)
        convert_directory(sys.argv[1], sys.argv[2])
        """)
        cmd = [sys.executable, "-c", helper, dicom_folder, temp_folder]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)

        if proc.returncode != 0:
            return f"dicom2nifti failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"

        print(f"Converted {dicom_folder} to NIfTI format in {temp_folder}")
        return None
    except subprocess.TimeoutExpired:
        return f"Timeout converting {dicom_folder}"
    except Exception as e:
        return f"Error converting {dicom_folder}: {e}"

def rename_and_move_nifti_files(temp_folder, main_output_folder, volume_id, patient_id, study_name):
    """
    Renames the most recent NIfTI file in the temp folder, then moves it to a structured patient/study folder.
    """
    patient_folder = os.path.join(main_output_folder, patient_id)
    study_folder = os.path.join(patient_folder, study_name)
    os.makedirs(study_folder, exist_ok=True)

    nifti_files = [f for f in os.listdir(temp_folder) if f.endswith(".nii.gz")]
    if not nifti_files:
        print(f"No NIfTI files found in {temp_folder}")
        # Clean up temp folder anyway
        shutil.rmtree(temp_folder, ignore_errors=True)
        return None

    latest_file = max(nifti_files, key=lambda f: os.path.getmtime(os.path.join(temp_folder, f)))
    latest_file_path = os.path.join(temp_folder, latest_file)

    new_filename = f"{volume_id}.nii.gz"
    new_filename = re.sub(r'[^a-zA-Z0-9_.]', '_', new_filename)
    new_filepath = os.path.join(study_folder, new_filename)

    shutil.move(latest_file_path, new_filepath)
    print(f"Moved and renamed {latest_file} to {new_filepath}")

    shutil.rmtree(temp_folder, ignore_errors=True)
    return new_filepath

def save_metadata_to_excel(df_ukbb, df_fastmri, output_file):
    """
    Save the complete metadata to an Excel file with just two sheets.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_ukbb.to_excel(writer, sheet_name='TD_patients', index=False)
        df_fastmri.to_excel(writer, sheet_name='fastMRI', index=False)
    print(f"Excel file saved to {output_file}")

def main():
    """Main workflow combining metadata extraction and filtering."""

    # Step 0: Ensure output directories exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(TD_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FASTMRI_OUTPUT_FOLDER, exist_ok=True)

    # Step 1: Extract metadata (Keep in memory)
    df_fastmri = pd.DataFrame(fastmri_importer(ROOT_FOLDER_FASTMRI))
    df_ukbb = pd.DataFrame(ukbb_importer(ROOT_FOLDER_TD_data))
    print('#Step1 Done')

    # Step 2: Check completeness - now adds 'selected_for_conversion' flag
    df_ukbb = select_volumes(df_ukbb, ACCEPTED_DESCRIPTIONS)
    df_fastmri = select_volumes(df_fastmri, ACCEPTED_DESCRIPTIONS)
    print('#Step2 Done')

    # Step 3: Only the 3 selected volumes per study ID
    ukbb_to_process = df_ukbb[df_ukbb['selected_for_conversion']].to_dict('records') if (df_ukbb is not None and not df_ukbb.empty) else []
    fastmri_to_process = df_fastmri[df_fastmri['selected_for_conversion']].to_dict('records')
    print('#Step3 Done')

    # Step 4: Convert & update dataframes
    df_ukbb = process_selected_dicom_folders(ukbb_to_process, TD_OUTPUT_FOLDER, 'TD_patients', df_ukbb)
    df_fastmri = process_selected_dicom_folders(fastmri_to_process, FASTMRI_OUTPUT_FOLDER, 'fastMRI', df_fastmri)
    print('#Step4 Done')

    # Step 5: Save the full dataset (with the flag) to Excel
    save_metadata_to_excel(df_ukbb, df_fastmri, OUTPUT_EXCEL_PATH)
    print("Processing complete. Results saved to:", OUTPUT_EXCEL_PATH)

     # Final cleanup of any remaining temp_conversion folders
    for root, dirs, _ in os.walk(OUTPUT_PATH):
        for d in dirs:
            if d.startswith("temp_conversion"):
                temp_path = os.path.join(root, d)
                try:
                    shutil.rmtree(temp_path, ignore_errors=True)
                    print(f"🧹 Cleaned up {temp_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove {temp_path}: {e}")

if __name__ == "__main__":
    main()
