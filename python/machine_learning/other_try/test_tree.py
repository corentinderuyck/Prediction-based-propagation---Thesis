import pandas as pd
import joblib
import numpy as np

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

def evaluate_model(csv_path, model_path, threshold=0.5):
    df = pd.read_csv(csv_path)

    model = joblib.load(model_path)

    X = df.drop(columns=["label"])
    y = df["label"].values.astype(int)

    probs = model.predict_proba(X)[:, 1]

    preds = (probs > threshold).astype(int)

    tp_0 = fp_0 = tn_0 = fn_0 = 0
    tp_1 = fp_1 = tn_1 = fn_1 = 0

    for pred, target in zip(preds, y):
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

evaluate_model("../data/tabular_features_test.csv", "../data/random_forest_model.joblib")
