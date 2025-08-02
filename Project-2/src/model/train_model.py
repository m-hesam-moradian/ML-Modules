from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from .evaluate import evaluate_model


def train_raw_model(model_name, X_train, y_train, X_test, y_test):
    params = []
    # Choose model
    if model_name == "ETR":
        # params=random_state=42, n_estimators=100, max_depth=3, min_samples_split=5
        params = {
            "random_state": 42,
            "n_estimators": 99,
            "max_depth": 3,
            "min_samples_split": 5,
        }

        model = ExtraTreesRegressor(
            random_state=params["random_state"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )  # Default parameters
    elif model_name == "LR":
        # params=random_state=42, alpha=0.1, max_iter=1000, tol=0.01
        params = {
            "random_state": 42,
            "alpha": 0.1,
            "max_iter": 1000,
            "tol": 0.01,
        }
        model = Lasso(
            random_state=params["random_state"],
            alpha=params["alpha"],
            max_iter=params["max_iter"],
            tol=params["tol"],
        )  # Default parameters
    else:
        raise ValueError("Unsupported model name. Use 'ETR' or 'LR'.")

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate MAE
    mae_error = mean_absolute_error(y_test, predictions)

    # Since we're using default params, params is just the default config

    return {
        "model_name": model_name,
        "predictions": predictions,
        "mae": mae_error,
        "params": params,
        "predictions": predictions,
    }


def get_X_y(df, target_col="Predicted Load (kW)"):
    """
    Splits the DataFrame into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_and_evaluate_models(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    etr_scores = []
    lasso_scores = []

    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\n🔁 Fold {fold} ------------------")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ✅ Extra Trees Regressor
        etr = ExtraTreesRegressor(
            random_state=42, n_estimators=100, max_depth=3, min_samples_split=5
        )
        etr.fit(X_train, y_train)
        etr_preds = etr.predict(X_val)
        etr_metrics = evaluate_model(y_val, etr_preds)
        etr_scores.append(etr_metrics)
        print(
            f"  ETR → R2: {etr_metrics['R2']:.4f}, RMSE: {etr_metrics['RMSE']:.4f}, MAE: {etr_metrics['MAE']:.4f}, MAPE: {etr_metrics['MAPE']:.2f}%, MARD: {etr_metrics['MARD']:.2f}%"
        )

        # ✅ Lasso Regression
        lasso = Lasso(random_state=42, alpha=0.11, max_iter=1000, tol=0.001)
        lasso.fit(X_train, y_train)
        lasso_preds = lasso.predict(X_val)
        lasso_metrics = evaluate_model(y_val, lasso_preds)
        lasso_scores.append(lasso_metrics)
        print(
            f"  Lasso → R2: {lasso_metrics['R2']:.4f}, RMSE: {lasso_metrics['RMSE']:.4f}, MAE: {lasso_metrics['MAE']:.4f}, MAPE: {lasso_metrics['MAPE']:.2f}%, MARD: {lasso_metrics['MARD']:.2f}%"
        )

        fold += 1

    return etr_scores, lasso_scores, kf


def get_best_fold_data_from_kf(X, y, scores, kf):
    """
    Uses provided KFold instance to return the train/test data
    for the fold with the highest R2 score.
    """
    best_fold_index = max(range(len(scores)), key=lambda i: scores[i]["R2"])

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        if i == best_fold_index:
            X_train_best = X.iloc[train_idx].copy()
            X_test_best = X.iloc[test_idx].copy()
            y_train_best = y.iloc[train_idx].copy()
            y_test_best = y.iloc[test_idx].copy()
            return X_train_best, X_test_best, y_train_best, y_test_best

    raise ValueError("Best fold index not found in KFold split.")
