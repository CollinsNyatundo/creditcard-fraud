# Data Directory — Raw Dataset

## Required File

This directory must contain the raw Kaggle Credit Card Fraud dataset **before**
any pipeline scripts can be run. The file is **not** included in the repository.

```
data/raw/creditcard.csv   ← place this file here
```

## How to Obtain the Data

1. Sign in to Kaggle at <https://www.kaggle.com/>.
2. Navigate to the dataset page:
   **Credit Card Fraud Detection** by ULB — Machine Learning Group
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
3. Click **Download** → download `creditcardfraud.zip`.
4. Extract the archive and place `creditcard.csv` in this directory:
   ```
   data/raw/creditcard.csv
   ```

## Dataset Description

| Property       | Value                                       |
|----------------|---------------------------------------------|
| Rows           | 284,807 transactions                        |
| Columns        | 31 (Time, V1–V28, Amount, Class)            |
| Fraud rate     | ~0.172% (492 fraudulent transactions)       |
| Feature origin | V1–V28 are anonymized PCA components        |
| Time period    | Two days of European card transactions      |
| Original paper | Dal Pozzolo et al., 2015 (Calibrating…)     |

## Citation

```bibtex
@misc{kaggle_creditcard_2016,
  title     = {Credit Card Fraud Detection},
  author    = {ULB Machine Learning Group},
  year      = {2016},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}
}
```

## License Notice

> **⚠️ License incompatibility warning**
>
> The Kaggle Credit Card Fraud dataset is distributed under the
> **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
> (CC BY-NC-SA 4.0)** license.
>
> This means:
> - The dataset may **not** be used for commercial purposes.
> - Any derivative work must be shared under the same license.
>
> The source code in this repository is licensed under **Apache 2.0**, which
> allows commercial use. However, **the dataset itself remains CC BY-NC-SA 4.0**
> and users must respect its non-commercial restriction.
>
> If you intend to use this pipeline commercially, you must replace the dataset
> with a compatible alternative (e.g., a synthetically generated dataset or one
> with a permissive license).

## Privacy Notes

The features V1–V28 are the result of a Principal Component Analysis (PCA)
transformation applied to the original transaction data. The original features
are **not disclosed** for confidentiality reasons. Only `Time` and `Amount`
retain their original semantics.

- **No PII (Personally Identifiable Information)** is present in the released
  dataset.
- The anonymization through PCA means re-identification risk is negligible.
- Standard GDPR / PCI DSS data-handling obligations apply if this pipeline is
  adapted to process real cardholder data in production.

## `.gitignore` note

`creditcard.csv` and `creditcardfraud.zip` are intentionally excluded from
version control (see `/.gitignore`). Only this `README.md` is tracked as a
placeholder to document data requirements.
