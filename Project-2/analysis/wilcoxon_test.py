# analysis/wilcoxon_test.py

from scipy.stats import wilcoxon

def perform_wilcoxon_test(mae_model_1, mae_model_2, model_name_1="Model 1", model_name_2="Model 2"):
    """
    Perform Wilcoxon signed-rank test between two model MAE lists.

    Args:
        mae_model_1 (list): MAE values for model 1 across folds.
        mae_model_2 (list): MAE values for model 2 across folds.
    """
    stat, p = wilcoxon(mae_model_1, mae_model_2)

    print(f"\n--- Wilcoxon Signed-Rank Test ---")
    print(f"Comparing: {model_name_1} vs {model_name_2}")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        print("✅ Significant difference at 95% confidence level.")
    else:
        print("❌ No significant difference.")
