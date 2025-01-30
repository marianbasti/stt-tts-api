#!/bin/bash

# Initialize and update submodules
git submodule init
git submodule update

# Create models directory if it doesn't exist
mkdir -p models

# Download the NEUROSYNC model from Hugging Face
echo "Downloading NEUROSYNC model..."
git lfs clone https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape tmp_model
mv tmp_model/* NeuroSync_Local_API/utils/model
rm -rf tmp_model

# Create symbolic links to NeuroSync utilities
ln -sf NeuroSync_Local_API/utils utils

echo "Setup complete! The NEUROSYNC model has been placed in utils/model"