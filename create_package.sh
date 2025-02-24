#!/bin/bash

# Create __init__.py files to make directories into Python packages
touch NeuroSync_Player/__init__.py
touch NeuroSync_Player/utils/__init__.py
touch NeuroSync_Player/utils/audio/__init__.py
touch NeuroSync_Player/utils/csv/__init__.py
touch NeuroSync_Player/livelink/__init__.py
touch NeuroSync_Player/livelink/animations/__init__.py
touch NeuroSync_Player/livelink/connect/__init__.py
touch NeuroSync_Player/generated/__init__.py

# Make the script executable again
chmod +x create_package.sh