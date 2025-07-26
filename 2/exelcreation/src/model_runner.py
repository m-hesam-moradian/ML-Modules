import importlib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_model(library, function, x_train, x_test, y_train, y_test, attributes=None):
    """
    Train and evaluate a scikit-learn model with comprehensive metrics,
    including predictions and true values.
    """
    if attributes is None:
        attributes = {}

    # Load model class dynamically
    sklearn_module = importlib.import_module(library)
    model_class = getattr(sklearn_module, function)
    model = model_class(**attributes)

    # Train and predict
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # Calculate metrics and store predictions and true labels
    metrics = {}
    for name, y_true, y_pred in [
        ("train", y_train, y_pred_train),
        ("test", y_test, y_pred_test),
        ("all", np.concatenate([y_train, y_test]), np.concatenate([y_pred_train, y_pred_test]))
    ]:
        metrics[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "predictions": y_pred,
            "true_values": y_true  
        }

        print(f"[{name.title():5}] "
              f"Accuracy: {metrics[name]['accuracy']:.4f} | "
              f"Precision: {metrics[name]['precision']:.4f} | "
              f"Recall: {metrics[name]['recall']:.4f} | "
              f"f1_score: {metrics[name]['f1_score']:.4f}")

    metrics["model"] = model
    return metrics
