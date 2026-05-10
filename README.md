# Movie Popularity Prediction Dashboard

Streamlit dashboard for testing movie popularity prediction models.

The project includes two ML tasks:

- Regression: predicts a continuous `popularity` score.
- Classification: predicts a `popularityLevel` class.

The dashboard loads saved preprocessing files and trained model `.pkl` files from the two milestone folders, runs predictions on the included test CSV files, and lets users upload their own CSV files.

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- milestone_1_linear_regression/
|   |-- test_script_m1.py
|   |-- test_data_m1.csv
|   |-- preprocessing_m1.pkl
|   `-- model_*_m1.pkl
|-- milestone_2_classification/
|   |-- test_script2.py
|   |-- test_data_m2.csv
|   |-- preprocessing.pkl
|   `-- model_*.pkl
`-- .streamlit/
    `-- config.toml
```

## Requirements

- Python 3.10 or newer recommended
- pip
- Git, if cloning from a repository

Python packages are listed in `requirements.txt`:

- pandas
- numpy
- scikit-learn
- xgboost
- streamlit

## Setup

From the project root:

```powershell
cd "C:\Users\amral\ML project 3"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Dashboard

Start the Streamlit app:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser.

## Using the App

1. Choose `Regression` or `Classification` in the sidebar.
2. Use the included default test dataset, or upload your own CSV.
3. Run predictions.
4. Review metrics, prediction previews, and downloadable result files.

For regression, the app expects the target column `popularity` if you want metrics.

For classification, the app expects the target column `popularityLevel` if you want metrics.

Predictions can still run without the target column, but evaluation metrics will not be available.

## Run Test Scripts Directly

Regression:

```powershell
cd milestone_1_linear_regression
python test_script_m1.py test_data_m1.csv
```

Classification:

```powershell
cd milestone_2_classification
python test_script2.py test_data_m2.csv
```

These scripts generate prediction CSV files inside their milestone folders.

## Important Files

Do not delete these files unless you are retraining or replacing the models:

- `milestone_1_linear_regression/preprocessing_m1.pkl`
- `milestone_1_linear_regression/model_lr_m1.pkl`
- `milestone_1_linear_regression/model_poly2_m1.pkl`
- `milestone_1_linear_regression/model_rf_m1.pkl`
- `milestone_1_linear_regression/model_xgb_m1.pkl`
- `milestone_2_classification/preprocessing.pkl`
- `milestone_2_classification/model_lr.pkl`
- `milestone_2_classification/model_rf.pkl`
- `milestone_2_classification/model_xgb.pkl`
- `milestone_2_classification/model_dt.pkl`

The app depends on these saved artifacts.

## Troubleshooting

If `streamlit` is not recognized:

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

If `xgboost` fails to install, upgrade pip first:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If model loading fails, confirm that all `.pkl` files are present in the milestone folders.

If uploaded CSV predictions fail, compare the uploaded CSV columns with the included test files:

- `milestone_1_linear_regression/test_data_m1.csv`
- `milestone_2_classification/test_data_m2.csv`

## Notes for Contributors

- Keep generated cache folders like `__pycache__/` out of Git.
- Keep local environments like `.venv/` out of Git.
- Do not commit private files such as `.env` or `.streamlit/secrets.toml`.
- Update this README when setup steps, dependencies, or model artifact names change.
