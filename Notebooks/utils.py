import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from pathlib import Path
from datetime import datetime


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    save_path="results",
    experiment_name="Unknown",
    explanation=None
):
    """
    Evaluate classification model and append results to a CSV file.

    Stores:
    - Train/Test Accuracy
    - Train/Test F1
    - Train/Test Precision
    - Train/Test Recall
    - ROC-AUC
    - Overfitting gap
    - Experiment metadata

    Returns
    -------
    DataFrame row of current experiment
    """

    # -------------------------
    # Create results folder
    # -------------------------
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)

    results_path = save_dir / "results.csv"

    # -------------------------
    # Predictions
    # -------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # -------------------------
    # ROC AUC (handle models without predict_proba)
    # -------------------------
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
    except:
        roc_auc = None

    # -------------------------
    # Model name
    # -------------------------
    model_name = model.__class__.__name__

    # -------------------------
    # Timestamp
    # -------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------
    # Build results row
    # -------------------------
    results_row = pd.DataFrame([{
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "explanation": explanation,
        "model": model_name,

        "Accuracy_Train": round(accuracy_score(y_train, y_train_pred), 4),
        "Accuracy_Test": round(accuracy_score(y_test, y_test_pred), 4),

        "F1_Train": round(f1_score(y_train, y_train_pred, average="weighted"), 4),
        "F1_Test": round(f1_score(y_test, y_test_pred, average="weighted"), 4),

        "Precision_Train": round(precision_score(y_train, y_train_pred, average="weighted"), 4),
        "Precision_Test": round(precision_score(y_test, y_test_pred, average="weighted"), 4),

        "Recall_Train": round(recall_score(y_train, y_train_pred, average="weighted"), 4),
        "Recall_Test": round(recall_score(y_test, y_test_pred, average="weighted"), 4),

        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,

        "Overfit_Gap": round(
            accuracy_score(y_train, y_train_pred) -
            accuracy_score(y_test, y_test_pred), 4
        )
    }])

    # -------------------------
    # Append to CSV (experiment history)
    # -------------------------
    if results_path.exists():
        old_results = pd.read_csv(results_path)
        results_df = pd.concat([old_results, results_row], ignore_index=True)
    else:
        results_df = results_row

    results_df.to_csv(results_path, index=False)

    return results_row