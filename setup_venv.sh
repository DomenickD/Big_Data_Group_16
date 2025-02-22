#!/bin/bash

# To run this script:
# 1. Save it as a .sh file (e.g., setup_venv.sh)
# 2. Make it executable: chmod +x setup_venv.sh
# 3. Run it: ./setup_venv.sh

VENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"

# Create and activate the virtual environment, then install requirements.
python3 -m venv "$VENV_NAME" && source "$VENV_NAME/bin/activate" && pip install -r "$REQUIREMENTS_FILE"

# Check if the venv already exists
if [ -d "$VENV_NAME" ]; then
  echo "Virtual environment '$VENV_NAME' already exists. Activating and updating requirements..."
  source "$VENV_NAME/bin/activate" && pip install --upgrade -r "$REQUIREMENTS_FILE"
else
  # Create and activate the virtual environment, then install requirements.
  python3 -m venv "$VENV_NAME" && source "$VENV_NAME/bin/activate" && pip install -r "$REQUIREMENTS_FILE"
fi

# Provide user feedback.
if [ $? -eq 0 ]; then
  echo "Virtual environment '$VENV_NAME' created and activated, and requirements installed."
  echo "You can now run your Python scripts."
else
  echo "An error occurred. Check the output."
  exit 1
fi

# type the following command in the terminal to deactivate the virtual environment:
# deactivate