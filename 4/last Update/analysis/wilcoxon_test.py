import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon


# Read the "PREDICTIONS" sheet
df = pd.read_excel("../data/Data.xlsx", sheet_name="PREDICTIONS")

# Optional: check the first few rows
# print(df.head())

# Ensure only numeric prediction columns are used
prediction_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Store results
results = []

# Perform Wilcoxon signed-rank test pairwise
for i in range(len(prediction_columns)):
    for j in range(i + 1, len(prediction_columns)):
        col1 = prediction_columns[i]
        col2 = prediction_columns[j]

        try:
            stat, p = wilcoxon(df[col1], df[col2])
            results.append(
                {"Comparison": f"{col1} vs {col2}", "Statistic": stat, "P-Value": p}
            )
        except ValueError as e:
            results.append(
                {
                    "Comparison": f"{col1} vs {col2}",
                    "Statistic": "Error",
                    "P-Value": str(e),
                }
            )

# Create results DataFrame
wilcoxon_df = pd.DataFrame(results)

# Display or save
print(wilcoxon_df)
# wilcoxon_df.to_csv("wilcoxon_results.csv", index=False)
