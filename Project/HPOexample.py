from sklearn.ensemble import RandomForestRegressor  # RFR
from sklearn.tree import DecisionTreeRegressor      # DTR
from sklearn.gaussian_process import GaussianProcessRegressor  # GPR
import pandas as pd

from src.Optimizer.HPO import HPO_runner

# === HPO Runner ===


if __name__ == "__main__":
    best_params, best_accuracy, convergence_curve = HPO_runner(
        file_path="data/357_Computer_Hardware_RFE_NBR_KNNR_TSOA_CO_SHAP_M_Nasirianfar_2.xlsx",
        target_column='PRP',
        model=RandomForestRegressor(),  # Model Object
        params={
            'n_estimators': [10, 500],  # Hyperparameters with ranges
            'max_depth':[1,50],
            'min_samples_leaf':[1,4],
            'min_samples_split':[2,10]
        },
        task_type="regression", 
        SearchAgents=5,
        Max_iterations=20,
        test_size=0.2
    )
#     print(convergence_curve)
# if __name__ == "__main__":
#     best_params, best_accuracy, convergence_curve = HPO_runner(
#         file_path="data/FraudDetectionDataset.xlsx",
#         target_column='Fraudulent',
#         model=RandomForestClassifier(),  # Model Object
#         params={
#             'n_estimators': [5, 50],  # Hyperparameters with ranges
#         },
#         task_type="classification", 
#         SearchAgents=5,
#         Max_iterations=5,
#         test_size=0.2
#     )