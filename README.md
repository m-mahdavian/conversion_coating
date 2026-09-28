# Conversion Coating XRD Analysis — README

## Overview
This repository provides Python scripts to process and analyze raw datasets for conversion coating studies. The workflow includes feature visualization (PCA on FTIR-derived features), model selection and evaluation, hyperparameter tuning of the best model, and visualization of experimental validation using unseen samples.

## Data Files
Place the following raw data files in your local project directory:
- `single_conversion_layer.xlsx`
- `epoxy_coated.xlsx`
- `XRD.xlsx`
- `XRD_validation.xlsx`
- `epoxy_coated_validation.xlsx`

## Prerequisites
- Python 3.8+  
- Recommended packages (install via `pip` as needed):
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `xgboost`
  
You can install the common dependencies with:
```bash
pip install numpy pandas scikit-learn matplotlib xgboost
```

## Setup
1. Clone or download this repository to your local machine.
2. Ensure the three data files listed above are available in the project directory.
3. In each script listed below, set the `project_path` variable to the absolute path of your local project directory. Preserve the path format used in the script.

## Workflow and Scripts

### 1) Model Selection and PCA Visualization
Script: `conversion_coating_XRD_model__selection.py`

- Set `project_path` to your local project directory.
- This script:
  - Visualizes Principal Component Analysis (PCA) for FTIR-based features.
  - Trains and evaluates a set of candidate models.
  - Compares model performance to identify top-performing approaches.

### 2) Hyperparameter Tuning
Script: `conversion_coating_XRD_hp_tunning.py`  

- Set `project_path` as above.
- This script performs hyperparameter tuning for the best-performing model identified in the previous step (Random Forest).

### 3) Experimental Validation Visualization
Script: `conversion_coating_XRD_visulalization.py`  

- Set `project_path` as above.
- This script visualizes results for the experimental validation stage using unseen samples.

## Notes
- Keep the original filenames of the scripts and data files to avoid import or path issues.
- Ensure that the directory structure and `project_path` match your local setup.
- If you encounter missing package errors, install the required packages listed in Prerequisites.

## Contact
For issues or questions, please open an issue in this repository.
