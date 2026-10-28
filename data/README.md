# Dataset Acquisition & Attribution (`data/`)

## Primary Benchmark Dataset: IBM Telco Customer Churn

**ChurnLens** is designed around the public **IBM Telco Customer Churn** dataset published by IBM Developer / IBM Sample Data:
- **Canonical Source Repository**: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Direct Raw CSV URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Alternative Public Mirror**: [Kaggle — Telco Customer Churn (`blastchar/telco-customer-churn`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Expected Dataset Characteristics (Official IBM Telco CSV)
- **Rows**: `7,043` customer records
- **Columns**: `21` columns (`1` identifier `customerID`, `19` customer/service/billing attributes, and `1` binary target `Churn`)
- **Target Variable (`Churn`)**: `"Yes"` encoded as `1` (positive class, `1,869` rows / `26.54%`) and `"No"` encoded as `0` (negative class, `5,174` rows / `73.46%`)
- **Known Data-Cleaning Edge Case**: The `TotalCharges` column contains `11` blank whitespace strings (`" "`) corresponding to new customers with `tenure == 0`. `ChurnLens` automatically coerces whitespace strings in `TotalCharges` to `NaN` via `pd.to_numeric(..., errors="coerce")` and imputes them inside the training-fitted `SimpleImputer(strategy="median")` pipeline so no test-set statistics leak during imputation.

## How to Download the Dataset

### Windows PowerShell / Command Prompt / Linux / macOS
From the `ChurnLens/` repository root:
```powershell
python scripts/download_data.py
```
This downloads `Telco-Customer-Churn.csv` into `data/raw/Telco-Customer-Churn.csv` and verifies its row count, column names, and SHA-256 checksum.

### Manual Placement (If Behind a Corporate/Campus Firewall)
Download `Telco-Customer-Churn.csv` from the URL above in your browser and save it to:
```text
ChurnLens/data/raw/Telco-Customer-Churn.csv
```

## Offline Synthetic Fixture Mode (`--use-synthetic`)
When outbound internet access is unavailable (such as in isolated CI containers or offline sandboxes), you can run the complete pipeline using the built-in 21-column synthetic Telco fixture (`generate_synthetic_telco_fixture` in `src/churnlens/data.py`), which mirrors the exact IBM Telco column schema, blank-string `TotalCharges` edge cases, and ~26.5% class imbalance:
```powershell
python scripts/run_experiment.py --config configs/default.yaml --use-synthetic --output-dir results/synthetic_verification
```
Outputs from synthetic fixture runs are explicitly labeled `"data_source": "synthetic_fixture"` in all metadata and tables so they are never confused with real IBM Telco benchmark results.

## Licensing & Redistribution Note
Raw CSV files inside `data/raw/` and `data/processed/` are excluded from Git via `.gitignore`. The MIT License in this repository covers the **ChurnLens** source code and documentation only; the IBM Telco dataset remains subject to [IBM's sample data terms](https://github.com/IBM/telco-customer-churn-on-icp4d).
