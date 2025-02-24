#!/bin/bash

# First remove any existing NeuroSync_Player directory if it exists
rm -rf NeuroSync_Player

# Initialize and update submodules
echo "Initializing submodules..."
git submodule add https://github.com/AnimaVR/NeuroSync_Player.git NeuroSync_Player
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
mkdir -p NeuroSync_Player/checkpoint
mv tmp_model/* NeuroSync_Player/checkpoint
rm -rf tmp_model

# Download the XTTS model from Hugging Face
echo "Downloading XTTS model..."
git lfs clone https://huggingface.co/marianbasti/XTTS-v2-argentinian-spanish models/XTTS-v2-argentinian-spanish

chmod +x create_package.sh
./create_package.sh

echo "Setup complete! Models have been placed in their respective directories"