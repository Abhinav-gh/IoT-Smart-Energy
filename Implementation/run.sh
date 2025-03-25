#!/bin/bash

echo "Running preprocessing..."
python ./Pre-Processing/preprocessing.py

echo "Running training..."
python ./K_fold_LGBM/train.py

echo "Pipeline completed!"
