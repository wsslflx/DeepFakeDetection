import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

with open("features_train.pkl", "rb") as f: train = pickle.load(f)
with open("features_val.pkl",   "rb") as f: val   = pickle.load(f)

X_train, y_train = train["X"], train["y"]
X_val,   y_val   = val["X"],   val["y"]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Train
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=10,
    tree_method="hist",
    random_state=42,
)
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Evaluate
preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"\nVal accuracy: {acc:.4f}")

# Per-class breakdown
NAMES = ["real","hmar_d20","hmar_d30","llamagen_B","llamagen_L",
         "var_d20","var_d30","rar_l","rar_xxl"]
cm = confusion_matrix(y_val, preds)
print("\nPer-class accuracy:")
for i, name in enumerate(NAMES):
    print(f"  {name:15s}: {cm[i,i]/cm[i].sum():.3f}")

# Save
import joblib
joblib.dump(clf, "xgb_model.pkl")
print("\nModel saved to xgb_model.pkl")