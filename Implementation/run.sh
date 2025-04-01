#!/bin/bash

# Ask the user to choose Python or Python3
read -p "Enter the Python interpreter to use (default: python): " PYTHON_INTERPRETER
PYTHON_INTERPRETER=${PYTHON_INTERPRETER:-python}  # Default to "python" if empty

# Help Option
if [ "$1" == "--help" ]; then
    echo "This is the shell script to run the ASHRAE Prediction Pipeline."
    echo "Usage: $0 [--faster | --help]"
    echo "  --faster   Run preprocessing on a reduced dataset for faster execution."
    echo "  --help     Show this help message and exit."
    exit 0
fi

echo "Starting ASHRAE Prediction Pipeline using '$PYTHON_INTERPRETER'..."

# Step 0: Prepare the dataset.
echo "Preparing Dataset..."
$PYTHON_INTERPRETER ./prepare_data.py

# Step 1: Preprocessing
echo "Running preprocessing..."

# Check if the user wants faster preprocessing
if [ "$1" == "--faster" ]; then
    echo "Running faster preprocessing on reduced dataset..."
    $PYTHON_INTERPRETER ./Pre-Processing/preprocessing.py --faster
else
    echo "Running full preprocessing..."
    $PYTHON_INTERPRETER ./Pre-Processing/preprocessing.py
fi

# Step 2: Training using K-fold LightGBM
echo "Running training..."
$PYTHON_INTERPRETER ./K_fold_LGBM/train.py

# Step 3: Inference
echo "Running inference..."
$PYTHON_INTERPRETER ./K_fold_LGBM/inference.py

echo "Pipeline completed successfully!"
echo "You can find the results in the 'Results_and_Plots' directory."