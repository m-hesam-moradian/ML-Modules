import pandas as pd
import numpy as np
import time
from src.data_loader import load_data
from src.feature_engineering import average_daily
from src.model.train_model import get_X_y
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


def couples_sensitivity_analysis(
    model, X, y, feature_pairs, metric="mse", perturbation=0.1
):
    """
    Perform sensitivity analysis for couples (pairs) of features in a machine learning model.

    Parameters:
    -----------
    model : trained machine learning model
        The trained model object with a `predict` method.

    X : pd.DataFrame or np.ndarray
        Input data for the model.

    y : pd.Series or np.ndarray
        True labels or target values.

    feature_pairs : list of tuples
        List of pairs of feature names or indices for sensitivity analysis.

    metric : str, optional, default='mse'
        Evaluation metric to measure the change in predictions.
        Options: 'mse', 'mae', 'accuracy'.

    perturbation : float, optional, default=0.1
        The percentage by which to perturb the feature values.

    Returns:
    --------
    pd.DataFrame
        Sensitivity report for each feature pair, showing the effect of perturbations.
    """

    # Define metric function
    if metric == "mse":
        metric_func = mean_squared_error
    elif metric == "mae":
        metric_func = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == "accuracy":
        metric_func = accuracy_score
    else:
        raise ValueError("Unsupported metric. Choose from 'mse', 'mae', 'accuracy'.")

    original_predictions = model.predict(X)
    original_score = metric_func(y, original_predictions)

    sensitivity_report = []

    for feature_1, feature_2 in feature_pairs:
        # Copy the data for perturbation
        X_perturbed = X.copy()

        # Perturb both features
        if isinstance(X_perturbed, pd.DataFrame):
            X_perturbed[feature_1] *= 1 + perturbation
            X_perturbed[feature_2] *= 1 + perturbation
        else:
            X_perturbed[:, feature_1] *= 1 + perturbation
            X_perturbed[:, feature_2] *= 1 + perturbation

        # Get new predictions
        perturbed_predictions = model.predict(X_perturbed)

        # Calculate score after perturbation
        perturbed_score = metric_func(y, perturbed_predictions)

        # Calculate sensitivity as the difference in metric
        sensitivity = perturbed_score - original_score

        # Store the results
        sensitivity_report.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "original_score": original_score,
                "perturbed_score": perturbed_score,
                "sensitivity": sensitivity,
            }
        )

    return pd.DataFrame(sensitivity_report)


# ⏱️ Start timer
start_timer = time.time()

# 📥 Load and preprocess data
DATA_PATH = "../data/Data.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="DATA after K-Fold")


X, y = get_X_y(df)

# 🎯 Create feature pairs for sensitivity analysis
features = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
from itertools import combinations_with_replacement

feature_pairs = list(combinations_with_replacement(features, 2))


# 🧠 Use tuned ETR model from main.py
model = ExtraTreesRegressor(
    random_state=int(79.864),
    n_estimators=int(37.1679),
    max_depth=int(6.7081),
    min_samples_split=int(2),
)
model.fit(X, y)

# 📊 Perform sensitivity analysis
sensitivity_df = couples_sensitivity_analysis(
    model=model, X=X, y=y, feature_pairs=feature_pairs, metric="mse", perturbation=0.1
)

# 📂 Save full sensitivity results (optional)
sensitivity_df.to_excel(
    "copula_sensitivity_full.xlsx", index=False, sheet_name="Sensitivity Full"
)


# 📈 Group by 'feature_1' and compute mean of numeric columns
grouped_data = sensitivity_df.groupby("feature_1")
mean_values_list = []

for name, group in grouped_data:
    numerical = group.select_dtypes(include=[np.number])
    mean_df = numerical.mean().to_frame().T
    mean_df["feature_1"] = name
    mean_values_list.append(mean_df)

# 🧮 Concatenate mean reports
copula_mean_df = pd.concat(mean_values_list, ignore_index=True)
copula_mean_df = copula_mean_df[
    ["feature_1"] + [col for col in copula_mean_df.columns if col != "feature_1"]
]

# 💾 Save summarized results
copula_mean_df.to_excel(
    "copula_sensitivity_summary.xlsx", index=False, sheet_name="Sensitivity Summary"
)

# 🕒 Print processing time
print(f"✅ Finished sensitivity analysis in {time.time() - start_timer:.2f} seconds.")
print(copula_mean_df.head())
