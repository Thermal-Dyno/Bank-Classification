"""
Binary Classification with a Bank Dataset (Kaggle Playground S5E8)
Goal: predict probability that a client subscribes to a term deposit (y = 1).
Metric: ROC AUC  -> submit probabilities, not 0/1 labels.

Install:
    pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn

Run:
    python bank_classification.py
Put train.csv, test.csv, sample_submission.csv in DATA_DIR (default: current folder).
"""

import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = "."          # change if your files are elsewhere
SEED = 42
N_FOLDS = 5
TARGET = "y"

# ---------------------------------------------------------------- 1. LOAD
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print("train", train.shape, "| test", test.shape)
print("positive rate:", round(train[TARGET].mean(), 4))

# ---------------------------------------------------------------- 2. FEATURES
CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
MONTH_MAP = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_num"] = df["month"].map(MONTH_MAP)
    df["day_of_year"] = df["month_num"] * 31 + df["day"]          # rough calendar position
    df["was_contacted"] = (df["pdays"] != -1).astype(int)
    df["balance_neg"] = (df["balance"] < 0).astype(int)
    df["log_balance"] = np.sign(df["balance"]) * np.log1p(df["balance"].abs())
    df["log_duration"] = np.log1p(df["duration"])
    df["duration_per_campaign"] = df["duration"] / (df["campaign"] + 1)
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["age_bin"] = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 120], labels=False)
    df["has_any_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)
    df["job_edu"] = df["job"] + "_" + df["education"]
    return df


X = add_features(train.drop(columns=[TARGET, "id"]))
y = train[TARGET].values
X_test = add_features(test.drop(columns=["id"]))

cat_features = CAT_COLS + ["job_edu"]
num_features = [c for c in X.columns if c not in cat_features]

# category dtype for LightGBM / XGBoost, string for CatBoost
X_cat, X_test_cat = X.copy(), X_test.copy()
for c in cat_features:
    cats = pd.Categorical(pd.concat([X[c], X_test[c]])).categories
    X_cat[c] = pd.Categorical(X[c], categories=cats)
    X_test_cat[c] = pd.Categorical(X_test[c], categories=cats)

# one-hot version for the logistic regression baseline
X_ohe = pd.get_dummies(pd.concat([X, X_test]), columns=cat_features, drop_first=True)
X_ohe_train, X_ohe_test = X_ohe.iloc[:len(X)], X_ohe.iloc[len(X):]

# ---------------------------------------------------------------- 3. CV HELPERS
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = {}   # name -> (oof_preds, test_preds)


def report(name, oof, tpred):
    results[name] = (oof, tpred)
    print(f"==> {name}: CV AUC = {roc_auc_score(y, oof):.5f}\n")


# ---------------------------------------------------------------- 4. BASELINE
print("Baseline: Logistic Regression")
oof, tpred = np.zeros(len(X)), np.zeros(len(X_test))
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
    m.fit(X_ohe_train.iloc[tr], y[tr])
    oof[va] = m.predict_proba(X_ohe_train.iloc[va])[:, 1]
    tpred += m.predict_proba(X_ohe_test)[:, 1] / N_FOLDS
report("logreg", oof, tpred)

# ---------------------------------------------------------------- 5. LIGHTGBM
print("LightGBM")
lgb_params = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
    min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, cat_smooth=20, verbose=-1, seed=SEED,
)
oof, tpred = np.zeros(len(X)), np.zeros(len(X_test))
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    dtr = lgb.Dataset(X_cat.iloc[tr], y[tr])
    dva = lgb.Dataset(X_cat.iloc[va], y[va], reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va] = m.predict(X_cat.iloc[va], num_iteration=m.best_iteration)
    tpred += m.predict(X_test_cat, num_iteration=m.best_iteration) / N_FOLDS
    print(f"  fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f} (trees {m.best_iteration})")
report("lightgbm", oof, tpred)

# ---------------------------------------------------------------- 6. XGBOOST
print("XGBoost")
oof, tpred = np.zeros(len(X)), np.zeros(len(X_test))
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    m = xgb.XGBClassifier(
        n_estimators=5000, learning_rate=0.03, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True, eval_metric="auc",
        early_stopping_rounds=150, random_state=SEED, n_jobs=-1,
    )
    m.fit(X_cat.iloc[tr], y[tr], eval_set=[(X_cat.iloc[va], y[va])], verbose=False)
    oof[va] = m.predict_proba(X_cat.iloc[va])[:, 1]
    tpred += m.predict_proba(X_test_cat)[:, 1] / N_FOLDS
    print(f"  fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f}")
report("xgboost", oof, tpred)

# ---------------------------------------------------------------- 7. CATBOOST
print("CatBoost")
oof, tpred = np.zeros(len(X)), np.zeros(len(X_test))
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    m = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=7, eval_metric="AUC",
        loss_function="Logloss", random_seed=SEED, od_type="Iter", od_wait=150,
        verbose=0, cat_features=cat_features,
        # task_type="GPU",   # uncomment on Kaggle/Colab with GPU
    )
    m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    tpred += m.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f}")
report("catboost", oof, tpred)

# ---------------------------------------------------------------- 8. ENSEMBLE
print("Ensemble (rank average of the three boosting models)")
from scipy.stats import rankdata

names = ["lightgbm", "xgboost", "catboost"]
rank = lambda a: rankdata(a) / len(a)
oof_ens = np.mean([rank(results[n][0]) for n in names], axis=0)
test_ens = np.mean([rank(results[n][1]) for n in names], axis=0)
print(f"==> Ensemble CV AUC = {roc_auc_score(y, oof_ens):.5f}")

# ---------------------------------------------------------------- 9. SUBMISSION
sub[TARGET] = test_ens
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv", sub.shape)
print(sub.head())