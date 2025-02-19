#!/bin/bash

# Initialize and update submodules
git submodule sync
git submodule init
git submodule update --init --recursive

# Create and initialize .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Virtual environment already exists."
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

# Download the NEUROSYNC model from Hugging Face
echo "Downloading NEUROSYNC model..."
git lfs clone https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape tmp_model
mv tmp_model/* NeuroSync_Player/checkpoint
rm -rf tmp_model

# Create symbolic links to NeuroSync utilities
ln -sf NeuroSync_Player checkpoint

chmod +x create_package.sh
./create_package.sh

# Install dependencies
echo "Setup complete! The NEUROSYNC model has been placed in utils/model"