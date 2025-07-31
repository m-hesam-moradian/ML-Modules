# D:\ML\Project-2\src\model\train_model.py
from src.model.train_model import train_and_evaluate_models , get_X_y, get_best_fold_data_from_kf
import pandas as pd


def K_Fold(X, y, n_splits=5):
    etr_scores, lasso_scores ,kf= train_and_evaluate_models(X, y, n_splits=n_splits)
    # get_best_fold_data_from_kf for each model
    etr_X_train_best, etr_X_test_best, etr_y_train_best, etr_y_test_best = get_best_fold_data_from_kf(X, y, etr_scores, kf)
    lasso_X_train_best, lasso_X_test_best, lasso_y_train_best, lasso_y_test_best = get_best_fold_data_from_kf(X, y, lasso_scores, kf)


    # etr_scores_r2 = np.array([item['R2'] for item in etr_scores])
    # lasso_scores_r2 = np.array([item['R2'] for item in lasso_scores])
    # etr_scores_rmse = np.array([item['RMSE'] for item in etr_scores])
    # lasso_scores_rmse = np.array([item['RMSE'] for item in lasso_scores])


    # Check if both ETR and Lasso got the same train and test splits
    if (
        etr_X_train_best.equals(lasso_X_train_best) and
        etr_X_test_best.equals(lasso_X_test_best) and
        etr_y_train_best.equals(lasso_y_train_best) and
        etr_y_test_best.equals(lasso_y_test_best)
    ):
        print("ETR and Lasso folds are identical. Using shared variables.")
        X_train_shared = etr_X_train_best
        X_test_shared = etr_X_test_best
        y_train_shared = etr_y_train_best
        y_test_shared = etr_y_test_best
    else:
        print("ETR and Lasso folds differ. Keeping them separate.")

    # Preserve original y column name
    y_col_name = y_train_shared.name if hasattr(y_train_shared, 'name') else 'y'

    # Combine features and target
    train_df = pd.concat([X_train_shared.copy(), y_train_shared.copy()], axis=1)
    test_df = pd.concat([X_test_shared.copy(), y_test_shared.copy()], axis=1)

    # Merge train and test vertically
    combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    print("✅ Combined DataFrame using original target column name:")
    print(combined_df.head())

    return (X_train_shared, X_test_shared, y_train_shared, y_test_shared, combined_df)