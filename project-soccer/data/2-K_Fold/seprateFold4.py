import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# === Load data ===
file_path = "D:/ML/project-soccer/data/85_Soccer_ETR_LGBR_COA_BWO.xlsx"
df = pd.read_excel(file_path, sheet_name="DATA after VIF")

y = df["markat value"]
X = df.drop(columns=["markat value"])

# === Manual fold selection ===
n_splits = 5
fold_size = len(X) // n_splits
i = 3  # 4th fold (index starts from 0)

# === Create train/test split from fold 4 ===
start = i * fold_size
end = (i + 1) * fold_size

X_test_final = X.iloc[start:end]
y_test_final = y.iloc[start:end]

X_train_final = pd.concat([X.iloc[:start], X.iloc[end:]], axis=0)
y_train_final = pd.concat([y.iloc[:start], y.iloc[end:]], axis=0)

# === Combine train and test sets ===
train_df = X_train_final.copy()
train_df["markat value"] = y_train_final

test_df = X_test_final.copy()
test_df["markat value"] = y_test_final

combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# === Save to Excel ===
output_path = "D:/ML/project-soccer/fold4_train_test_combined.xlsx"
combined_df.to_excel(output_path, index=False)
print(f"Saved combined train + test to: {output_path}")


