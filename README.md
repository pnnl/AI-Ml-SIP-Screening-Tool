# SIP ML Screening Tool

## Overview

The SIP ML Screening Tool is a Python/Jupyter workflow for quality-control screening of spectral induced polarization (SIP) datasets. The tool combines CatBoost machine-learning classification with physics-guided screening rules, including spectral-shape checks and Kramers-Kronig consistency diagnostics, to assign spectra to three final categories:

- `Good`: spectra accepted for interpretation.
- `Review`: spectra that are uncertain, borderline, or require manual inspection.
- `Bad`: spectra with strong evidence of poor quality, artifacts, or physically implausible behavior.

The workflow is intended to support rapid and consistent pre-screening of large SIP datasets before interpretation or time-lapse comparison.

## Repository Contents

The deliverable package should contain the following required files and folders:

```text
SIP_ML_Tool.ipynb
requirements.txt
sip_ml_tool/
Raw SIP Data/
```

The main user-facing workflow is `SIP_ML_Tool.ipynb`. The detailed implementation is organized in the `sip_ml_tool/` Python modules so that the notebook remains readable and easier to review.

The `Final Results/` folder is created automatically when the workflow is run.

## Installation

Python 3.12 is recommended.

Install the required Python packages from the package folder:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- CatBoost
- python-pptx
- Jupyter
- nbconvert

## Required Input Folder Structure

The notebook uses relative paths and should be run from the package folder that contains `SIP_ML_Tool.ipynb`, `sip_ml_tool/`, and `Raw SIP Data/`.

The expected input folder structure is:

```text
Raw SIP Data/
+-- HPE SIP Data/
    +-- Good data/
    +-- Bad data/
    +-- Cautious data/
    +-- Unknown data/
```

The `Good data/` and `Bad data/` folders contain labeled spectra used for model training and grouped validation. The `Cautious data/` and `Unknown data/` folders contain spectra that are screened by the trained workflow.

## How To Run

Open `SIP_ML_Tool.ipynb` in Jupyter Notebook or JupyterLab and run the notebook cells from top to bottom.

The workflow is organized into two main stages:

- `run_training_stage()` loads the raw SIP files, builds the feature table, performs grouped validation, trains the CatBoost model, selects the physics-guided policy, and saves model artifacts.
- `run_prediction_stage()` reloads the saved artifacts, classifies all spectra, applies the physics-guided screening layer, and writes the final outputs.

For a full fresh run, execute both stages. For a faster prediction-only rerun after training artifacts already exist, run only `run_prediction_stage(...)`.

## Main Outputs

Primary outputs are written to `Final Results/`.

The main reader-facing files are:

```text
Final Results/01_README.md
Final Results/02_QC_Summary_Report.md
Final Results/03_QC_Proof_Scorecard.csv
Final Results/04_Final_Class_Counts.csv
Final Results/05_Final_Decision_Queue.csv
Final Results/06_Manual_Review_Queue.csv
Final Results/07_Gallery_GOOD.pptx
Final Results/08_Gallery_REVIEW.pptx
Final Results/09_Gallery_BAD.pptx
```

Detailed technical outputs are written to:

```text
Final Results/Validation/
```

This folder contains supporting validation tables, grouped out-of-fold predictions, feature tables, model artifacts, discovery audits, and traceability files.

## Interpreting Final Classifications

The final classification is not only the CatBoost prediction. It is determined by the CatBoost probability together with physics-guided screening rules.

`Good` means that the spectrum had sufficient model support and passed the required physical and spectral-shape checks.

`Review` means that the spectrum was not confidently accepted or rejected. These spectra should be inspected manually before interpretation.

`Bad` means that the spectrum showed strong evidence of poor quality, artifacts, or physically implausible behavior.

This structure is intentionally conservative. The workflow is designed to reduce the risk of automatically accepting poor-quality SIP spectra, while retaining uncertain spectra for review rather than forcing every spectrum into a binary Good/Bad decision.

## Reproducibility Notes

Results depend on the input data, folder structure, labeled training examples, and package versions listed in `requirements.txt`.

The workflow uses grouped validation for labeled spectra so that related spectra from the same experimental group are held together during model evaluation.

If the labeled data folders are changed, the training results, selected policy, and final classifications may also change.

## Citation

If using this tool, please cite the associated manuscript when available. Until publication, please contact the authors for citation information.

## Contact

Klaudio Peshtani  
Pacific Northwest National Laboratory  
klaudio.peshtani@pnnl.gov
