# Contributing to ChurnLens

Thank you for your interest in contributing to **ChurnLens**.

## Core Engineering Standards

1. **Zero Data Leakage**: All imputation (`SimpleImputer`), numerical scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) must remain inside a `scikit-learn` `ColumnTransformer` and `Pipeline` fitted strictly on the training partition.
2. **Untouched Test Set**: Any classification threshold tuning or model comparison decision must use a validation split derived solely from the training partition (`val_size_from_train`), never the held-out 20% test set.
3. **Offline Unit Tests**: All pytest tests in `tests/` must run on small in-memory synthetic fixtures without network access or external dataset downloads.
4. **Honest Reporting**: Never commit fabricated dataset statistics, invented accuracy numbers, or unverified business savings claims.

## Verification Before Submitting Changes

```powershell
# Windows PowerShell
python -m pytest tests -v
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```
