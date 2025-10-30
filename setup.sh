#!/usr/bin/env bash
# setup.sh — sets up all required micromamba environments

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the environments folder
cd "env/" || {
  echo "Error: Could not find SR-3D-pln/env directory."
  exit 1
}

# Create environments using micromamba
echo "Creating environment from env_step0_preprocess.yml..."
micromamba create -y -f env_step0_preprocess.yml

echo "Creating environment from env_step1_INR.yml..."
micromamba create -y -f env_step1_INR.yml

micromamba activate env_step1_INR
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
micromamba activate base

echo "Creating environment from env_step2_seg.yml..."
micromamba create -y -f env_step2_seg.yml

echo "Creating environment from env_step3_prepro.yml..."
micromamba create -y -f env_step3_prepro.yml

echo "Creating environment from env_step3_cWDM.yml..."
micromamba create -y -f env_step3_cWDM.yml

echo "✅ All environments have been successfully created!"
