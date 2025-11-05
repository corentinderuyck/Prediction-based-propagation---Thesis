import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def get_metrics(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1):
    total_samples = tp_1 + fp_1 + tn_1 + fn_1
    accuracy_global = (tp_1 + tn_1) / total_samples if total_samples > 0 else 0.0

    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    support_0 = tp_0 + fn_0

    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0.0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    support_1 = tp_1 + fn_1

    return {
        'accuracy_global': accuracy_global,
        'class_0': {'precision': precision_0, 'recall': recall_0, 'f1': f1_0, 'support': support_0},
        'class_1': {'precision': precision_1, 'recall': recall_1, 'f1': f1_1, 'support': support_1}
    }

print("Loading data...", flush=True)
df = pd.read_csv("../data/tabular_features.csv")

print("Balancing classes to 50/50 and limiting to 1 million examples...", flush=True)
n_per_class = 500_000

df_0 = df[df["label"] == 0]
df_1 = df[df["label"] == 1]

n_0 = len(df_0)
n_1 = len(df_1)

if n_0 < n_per_class or n_1 < n_per_class:
    n_per_class = min(n_0, n_1)
    print(f"Moins de 500k dans une des classes. Utilisation de {n_per_class} exemples par classe.")

df_balanced = pd.concat([
    df_0.sample(n=n_per_class, random_state=42),
    df_1.sample(n=n_per_class, random_state=42)
]).sample(frac=1, random_state=42)

X = df_balanced.drop(columns=["label"])
y = df_balanced["label"]

print("Splitting data...", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training model...", flush=True)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Saving model...", flush=True)
joblib.dump(model, "../data/random_forest_model.joblib")

print("Evaluating model...", flush=True)
probs = model.predict_proba(X_test)[:, 1]

threshold = 0.5
preds = (probs > threshold).astype(int)
targets = y_test.values.astype(int)

tp_0 = fp_0 = tn_0 = fn_0 = 0
tp_1 = fp_1 = tn_1 = fn_1 = 0

for pred, target in zip(preds, targets):
    pred_bool = bool(pred)
    target_bool = bool(target)

    tp_1 += int(pred_bool and target_bool)
    fp_1 += int(pred_bool and not target_bool)
    tn_1 += int(not pred_bool and not target_bool)
    fn_1 += int(not pred_bool and target_bool)

    tp_0 += int(not pred_bool and not target_bool)
    fp_0 += int(not pred_bool and target_bool)
    tn_0 += int(pred_bool and target_bool)
    fn_0 += int(pred_bool and not target_bool)

metrics = get_metrics(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1)
print(metrics)
