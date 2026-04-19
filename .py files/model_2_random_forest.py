# model_2_random_forest.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


DATA_PATH = "final_dataset.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "is_delayed"

from settings import DATA_PATH, TARGET, FEATURES

state_cols    = [c for c in df.columns if c.startswith("customer_state_") or c.startswith("seller_state_")]
category_cols = [c for c in df.columns if c.startswith("category_group_")]
ALL_FEATURES  = FEATURES + state_cols + category_cols

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()
bool_cols = X.select_dtypes(bool).columns
X[bool_cols] = X[bool_cols].astype(int)

print(f"Rows     : {df.shape[0]:,}")
print(f"Features : {len(ALL_FEATURES)}")
print(f"Delay %  : {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("\nModel trained!")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}  <- main metric")

cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"\n5-Fold CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

feat_imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances:")
print(feat_imp.round(4).to_string())

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["On-Time", "Delayed"],
    cmap="Greens"
)
plt.title("Random Forest — Confusion Matrix (Checkpoint 2)")
plt.tight_layout()
plt.savefig("c2_rf_confusion_matrix.png")
plt.show()
print("Plot saved!")
