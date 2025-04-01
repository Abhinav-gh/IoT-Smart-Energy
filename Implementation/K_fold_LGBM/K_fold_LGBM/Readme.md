# Directory Overview



This directory contains all the necessary files related to the training and inference of LightGBM models. Below is a breakdown of the key files and their purposes:

### 1. LightGBM Models
- The trained LightGBM models will be saved in this directory.
- There will be `k` saved model files (`lgbm_model_fold{k}.txt` format), where `k` is the number of folds used in cross-validation.
- Each model file corresponds to a different fold and will be used for evaluation.

### 2. Model Metadata
- A `model_info.json` file is created and stored in this directory.
- This file contains metadata about the trained models, including:
  - Number of folds (`k`).
  - Best-performing fold.
  - Model hyperparameters.
  - Paths to the saved models.

### 3. Test Dataset
- The processed test dataset is stored as `test_split.feather`.
- This dataset is unseen by the model and is used during inference to generate predictions.
- It contains the same features as the training dataset. Testing is done and evaluation metric used against the ground truth meter reading available in test dataset.
  

### 4. Running Inference
- Once training is complete, the inference script should be executed.

```python
python inference.py
```

- The script will:
  - Load the best-performing LightGBM model (as identified in `model_info.json`).
  - Load the test dataset (`test_split.feather`).
  - Perform predictions on the test set.
  - Save the predicted results for further evaluation or submission.

Ensure that all the required files are present before running inference to avoid errors.
