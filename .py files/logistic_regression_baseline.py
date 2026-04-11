import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


df = pd.read_csv(r"C:\Users\sanmu\Downloads\final_poc_dataset.csv")
print(f"Rows    : {df.shape[0]:,}")
print(f"Delay % : {df['is_delayed'].mean()*100:.1f}%")

FEATURES = [
    "approval_delay",
    "estimated_delivery_time",
    "purchase_day_of_week",
    "purchase_hour",
    "total_items",
    "total_price",
    "total_freight_value"
]

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
print("Model trained")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}  <- main metric")


