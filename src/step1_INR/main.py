import pdb
from tkinter.constants import HIDDEN
import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import os
from typing import Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from dataset import MultiDataset, InferDataset
from network.inr import WIREMLP, ReLUMLP
from network.pos_encoders import FourierFeatPosEncoder
import multiprocessing as mp
from pathlib import Path
from utils import scan_local_dataset_oriented, build_patient_study_list


if __name__ == '__main__':
    # Set the CUDA device 
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    mp.set_start_method('spawn', force=True)
    
    # Point to your local dataset root
    BASE_DIR = Path('../../data/00_sorted_nifti_for_INR/fastMRI_nifti')

    # Create the directory to store the results
    # Define the data root (the parent of sorted_nifti_for_INR)
    DATA_ROOT = BASE_DIR.parents[2]  # ../../data
    # Create the INR directory under data
    INR_dir = DATA_ROOT / 'data/01_INR'
    os.makedirs(INR_dir, exist_ok=True)
    # Determine the correct subfolder name
    if 'fastMRI' in str(BASE_DIR):
        subfolder = 'fastMRI_INR'
    elif 'TD_data' in str(BASE_DIR):
        subfolder = 'TD_data_INR'
    else:
        raise ValueError("BASE_DIR must contain either 'fastMRI' or 'TD_data'")
    # Create the full output path
    output_dir = INR_dir / subfolder
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be stored in: {output_dir}")

    subset_start, subset_end = 0, 100  # or None to process all in one go

    # SETTINGS
    batch_size = 2048 * 2
    HIDDEN_DIM = 1024
    NUM_LAYERS = 4
    NUM_EPOCHS = 100 # 1 for testing
    WIRE_OMEGA = 10.
    WIRE_SIGMA = 20.
    pos_encoder = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolution = [256, 256, 256]

    #### Create patient_study_list to run several INR's after each other ##############
    patient_study_map = scan_local_dataset_oriented(BASE_DIR)
    complete_patient_study_map = {}
    for patient_id, studies in patient_study_map.items():
        # Keep only studies that have all three orientations
        full_studies = {
            study_id: planes
            for study_id, planes in studies.items()
            if {'axial', 'coronal', 'sagittal'}.issubset(planes.keys())
        }
        if full_studies:
            complete_patient_study_map[patient_id] = full_studies
    print(f"Found {len(complete_patient_study_map)} patients with at least all 3 oriented files.")

    patient_study_list = build_patient_study_list(complete_patient_study_map, subset_start, subset_end)
    print(f"Selected {len(patient_study_list)} patient entries for processing.")
    ##################################################################################

    processed = 0
    for patient_id, study_ids in patient_study_list:
        for study_id in study_ids:
            

            planes = patient_study_map[patient_id][study_id]
            ax_path = planes.get('axial')
            cor_path = planes.get('coronal')
            sag_path = planes.get('sagittal')

            if not (ax_path and cor_path and sag_path):
                print(f"Skipping {patient_id}/{study_id} (missing one or more orientations). "
                      f"Have: AX={bool(ax_path)} COR={bool(cor_path)} SAG={bool(sag_path)}")
                continue


            dataset = MultiDataset(ax_path, cor_path, sag_path)

            print(f"Train with {batch_size} points per step on {patient_id}/{study_id}.")
            print(f"   AX: {Path(ax_path).name}")
            print(f"   COR: {Path(cor_path).name}")
            print(f"   SAG: {Path(sag_path).name}")
            print(torch.__version__)
            if torch.cuda.is_available():
                print("CUDA is available.")
            else:
                print("CUDA is not available. Using CPU.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Device: {device}')

            # Example to create a subdirectory for each run using time or a custom name
            run_name = time.strftime("run_%Y%m%d_%H%M%S")  # Timestamp-based unique run name
            log_dir = os.path.join('tb_logs', run_name)  # Set directory path
            os.makedirs(log_dir, exist_ok=True)

            logger = SummaryWriter(log_dir=log_dir)

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

            inr = WIREMLP(in_size=3,
                        out_size=3,
                        hidden_size=HIDDEN_DIM,
                        num_layers=NUM_LAYERS,
                        wire_omega=WIRE_OMEGA,
                        wire_sigma=WIRE_SIGMA).to(device)
            optimizer = torch.optim.Adam(inr.parameters(), lr=4e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)

            @torch.no_grad()
            def sample_at_resolution(model, resolution: Tuple[int, ...], batch_size: int = 4096, pos_encoder=None):
                """Evaluate our INR on a grid of coordinates in batches in order to obtain an image."""
                # Create the meshgrid and normalize coordinates

                x = torch.linspace(dataset.min_values[0], dataset.max_values[0], resolution[0])
                y = torch.linspace(dataset.min_values[1], dataset.max_values[1], resolution[1])
                z = torch.linspace(dataset.min_values[2], dataset.max_values[2], resolution[2])

                grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
                coords_norm_ = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1).to(device)

                # Prepare an empty list to hold batched predictions
                predictions_list = []

                num_coords = coords_norm_.shape[0]
                for i in range(0, num_coords, batch_size):
                    batch = coords_norm_[i:i + batch_size]
                    if pos_encoder:
                        batch = pos_encoder.forward(batch)

                    predictions_batch = model.forward(batch)  # Forward pass on a batch of coordinates
                    predictions_list.append(predictions_batch)

                # Stack all batched predictions and reshape to original resolution
                predictions_ = torch.cat(predictions_list, dim=0)
                predictions_0 = predictions_[:,0].reshape(resolution)
                predictions_1 = predictions_[:,1].reshape(resolution)
                predictions_2 = predictions_[:,2].reshape(resolution)

                return predictions_0, predictions_1, predictions_2


            for epoch in tqdm(range(NUM_EPOCHS+1)):
                for i, data in enumerate(tqdm(dataloader)):
                    inr.train()
                    optimizer.zero_grad()
                    coords, values = data   # Dataloader needs to output coords normalized to [-1, 1] and the intensity value
                    coords = coords.to(device)
                    values = values.to(device)

                    ax_mask = (values[:, 2] != -1.0)
                    values_ax = values[ax_mask, 2]
                    values_ax = values_ax.reshape(-1, 1)
                    coords_ax = coords[ax_mask, :]

                    cor_mask = (values[:, 1] != -1.0)
                    values_cor = values[cor_mask, 1]
                    values_cor = values_cor.reshape(-1, 1)
                    coords_cor = coords[cor_mask, :]

                    sag_mask = (values[:, 0] != -1.0)
                    values_sag = values[sag_mask, 0]
                    values_sag = values_sag.reshape(-1, 1)
                    coords_sag = coords[sag_mask, :]

                    coords = torch.cat((coords_ax, coords_cor, coords_sag), dim=0)
                    values = torch.cat((values_ax, values_cor, values_sag), dim=0)

                    if pos_encoder:
                        coords = pos_encoder.forward(coords)

                    out = inr.forward(coords)
                    out_ax = out[:len(coords_ax), 0:1]
                    out_cor = out[len(coords_ax):len(coords_ax)+len(coords_cor), 1:2]
                    out_sag = out[len(coords_ax)+len(coords_cor):, 2:3]

                    mi_out_ax = out[:, 0:1]
                    mi_out_cor = out[:, 1:2]
                    mi_out_sag = out[:, 2:3]

                    loss = (nn.functional.mse_loss(out_ax, values_ax) +
                            nn.functional.mse_loss(out_cor, values_cor) +
                            nn.functional.mse_loss(out_sag, values_sag))

                    loss.backward()
                    optimizer.step()

                    logger.add_scalar('loss', loss.item(), epoch + i + epoch * len(dataloader))
                scheduler.step()

                if epoch == NUM_EPOCHS:
                    print(f"Sample at epoch {epoch}")
                    pred_0, pred_1, pred_2 = sample_at_resolution(inr, resolution=resolution,
                                                batch_size=batch_size, pos_encoder=pos_encoder)

                    # Retrieve the affine matrix (assuming your dataset provides it)
                    affine_matrix = dataset.get_affine()
                    original_shape = dataset.get_dim()
                    print(f"Original shape {original_shape}")

                    # Compute target affine transformation
                    scaling_factors = torch.tensor([
                        original_shape[0] / resolution[1],
                        original_shape[1] / resolution[2],
                        original_shape[2] / resolution[0],
                        1  # Homogeneous coordinate remains 1
                    ], dtype=affine_matrix.dtype, device=affine_matrix.device)
                    print(f"Scaling factors {scaling_factors}")
                    # Create a diagonal matrix
                    scaling_matrix = torch.diag(scaling_factors)
                    # Compute the target affine
                    target_affine = affine_matrix.mm(scaling_matrix)

                    # Swap X and Y axes for each prediction
                    pred_0 = pred_0.flip(0)  # Swap Y and X axes
                    pred_1 = pred_1.flip(0)  # Swap Y and X axes
                    pred_2 = pred_2.flip(0)  # Swap Y and X axes

                    #Denormalization (mapping back)
                    min_values, max_values = dataset.get_minmax_values_original()

                    # Convert lists to tensors if max_values and min_values are lists
                    max_values = torch.tensor(max_values).to(device)
                    min_values = torch.tensor(min_values).to(device)

                    ###entnomalize here:
                    # Scale the predictions from normalized [-1, 1] back to the original range
                    pred_0 = (pred_0 + 1) * 0.5 * (max_values[0] - min_values[0]) + min_values[0]
                    pred_1 = (pred_1 + 1) * 0.5 * (max_values[1] - min_values[1]) + min_values[1]
                    pred_2 = (pred_2 + 1) * 0.5 * (max_values[2] - min_values[2]) + min_values[2]

                    #pdb.set_trace()
                    nib_img = Nifti1Image(pred_0.detach().cpu().numpy(), target_affine)
                    nib.save(nib_img, os.path.join(output_dir, f'{patient_id}_{study_id}_INR.nii.gz'))
                    #nib_img = Nifti1Image(pred_1.detach().cpu().numpy(), target_affine)
                    #nib.save(nib_img, f'./images/{patient_folder}_res_{output_resolution}_epoch_{epoch}_contr_1.nii.gz')
                    #nib_img = Nifti1Image(pred_2.detach().cpu().numpy(), target_affine)
                    #nib.save(nib_img, f'./images/{patient_folder}_res_{output_resolution}_epoch_{epoch}_contr_2.nii.gz')
                    print(f"Sampled for at resolution: {resolution}")
                    processed += 1
    print(f"Prepared {processed} study triplets for training.")        