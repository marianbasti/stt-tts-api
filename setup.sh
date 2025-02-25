#!/bin/bash

# First remove any existing NeuroSync_Player directory if it exists
if [ -d "NeuroSync_Player" ]; then
    echo "Removing existing NeuroSync_Player directory..."
    rm -rf NeuroSync_Player
    # Also remove it from .gitmodules if it exists
    git config --file=.gitmodules --remove-section "submodule.NeuroSync_Player" 2>/dev/null
    git add .gitmodules
    # Remove from .git/config
    git config --remove-section "submodule.NeuroSync_Player" 2>/dev/null
    # Clean up git cache
    git rm --cached NeuroSync_Player 2>/dev/null
    # Remove any leftover git index entries
    rm -f .git/modules/NeuroSync_Player 2>/dev/null
fi

# Initialize and update submodules
echo "Initializing submodules..."
git submodule add --force https://github.com/AnimaVR/NeuroSync_Player.git NeuroSync_Player
git submodule update --init --recursive --force

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

# Create model directory and copy files preserving structure
mkdir -p models/NEUROSYNC_Audio_To_Face_Blendshape
cp -r tmp_model/* models/NEUROSYNC_Audio_To_Face_Blendshape/

# Create a utils symlink at the top level for the NEUROSYNC model's imports to work
ln -sf models/NEUROSYNC_Audio_To_Face_Blendshape/utils utils

# Clean up tmp directory
rm -rf tmp_model

# Download the XTTS model from Hugging Face
echo "Downloading XTTS model..."
git lfs clone https://huggingface.co/marianbasti/XTTS-v2-argentinian-spanish models/XTTS-v2-argentinian-spanish

chmod +x create_package.sh
./create_package.sh

echo "Setup complete! Models have been placed in their respective directories"m