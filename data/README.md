# Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

## Data Setup

Download the IEEE-CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data):

1. Create a `data/` folder in this project root (if not exists)
2. Extract the following files into `data/`:
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv`
   - `test_identity.csv`

Alternatively, set the `IEEE_FRAUD_DATA_DIR` environment variable:
```bash
export IEEE_FRAUD_DATA_DIR="/path/to/ieee-fraud-detection"
```

## Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train_transaction.csv` | 590,540 | 394 | Transaction features (amounts, card, product, V-features) |
| `train_identity.csv` | 144,233 | 41 | Device and network identity features |
| `test_transaction.csv` | 506,691 | 393 | Test transactions (no label) |
| `test_identity.csv` | 141,907 | 41 | Test identity data |

## Dataset Details

- **Target**: `isFraud` (binary: 0=legitimate, 1=fraud)
- **Fraud rate**: ~3.5% (highly imbalanced class distribution)
- **Join key**: `TransactionID` (used for LEFT JOIN between transaction and identity tables)
- **V-features**: V1–V339 are anonymized proprietary features provided by Vesta

## Do Not Commit

The raw CSV files (~2GB) should **not** be committed to version control.
Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
