
# %%
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from src.Optimizer.HPO import HPO_runner
# === HPO Runner ===


if __name__ == "__main__":
    best_params, best_accuracy, convergence_curve = HPO_runner(
        file_path="data/357_Computer_Hardware_RFE_NBR_KNNR_TSOA_CO_SHAP_M_Nasirianfar_2.xlsx",
        target_column='PRP',
        model=RandomForestRegressor(),  # Model Object
        params={
            'n_estimators': [5, 50],  # Hyperparameters with ranges
        },
        task_type="regression", 
        SearchAgents=10,
        Max_iterations=10,
        test_size=0.5
    )







