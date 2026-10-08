# Bank Classification

## Midsem Project — Binary Classification with a Bank Dataset

## Project Objective

The objective of this project is to build machine learning models to predict whether a bank client will subscribe to a term deposit.

The competition uses **ROC AUC** as the evaluation metric. Therefore, the models generate probability predictions rather than binary class labels.

## Dataset

- **Training samples:** 750,000
- **Test samples:** 250,000
- **Target variable:** `y`

The dataset contains numerical and categorical banking-related features.

## Feature Engineering

The following additional features were created:

- `month_num`
- `day_of_year`
- `was_contacted`
- `balance_neg`
- `log_balance`
- `log_duration`
- `duration_per_campaign`
- `total_contacts`
- `age_bin`
- `has_any_loan`
- `job_edu`

## Models Used

Four approaches were evaluated:

1. Logistic Regression — baseline
2. LightGBM
3. XGBoost
4. CatBoost

The boosting models were evaluated using **5-fold Stratified Cross-Validation**.

## Model Performance

| Model | CV ROC AUC |
|---|---:|
| Logistic Regression | 0.94461 |
| LightGBM | 0.96895 |
| XGBoost | 0.96796 |
| CatBoost | 0.96814 |
| **Rank Ensemble** | **0.96899** |

## Ensemble

The final prediction uses a **rank average ensemble** of:

- LightGBM
- XGBoost
- CatBoost

The ensemble achieved a **CV ROC AUC of 0.96899**, slightly outperforming the best individual model, LightGBM.

## Submission

The final predictions are stored in:

`submission.csv`

The file contains **250,000 rows** with the columns:

```text
id
y
