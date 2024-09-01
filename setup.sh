#!/bin/bash

# Exit the script if any command fails
set -e

# Update
sudo apt-get update
sudo apt-get install libgl1-mesa-glx


# Upgrade pip to the latest version
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install --upgrade -r requirements.txt

# Inform the user that setup is complete
echo "Setup complete. Your environment is ready to use."

# Optionally, run tests to ensure everything is working
# echo "Running tests..."
# pytest


