import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

xls = pd.ExcelFile("data/85_Soccer_ETR_LGBR_COA_BWO.xlsx")
df = pd.read_excel("data/85_Soccer_ETR_LGBR_COA_BWO.xlsx", sheet_name="DATA")

y = df["markat value"]
X = df.drop(columns=["markat value"])

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def iterative_vif(X):
    X_current = X.copy()
    tables = []

    while X_current.shape[1] > 1:
        vif = calculate_vif(X_current)
        tables.append(vif.copy())
        max_vif_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
        print(f"Removing '{max_vif_feature}'")
        X_current = X_current.drop(columns=[max_vif_feature])

    if X_current.shape[1] == 1:
        last_col = X_current.columns[0]
        final_vif = pd.DataFrame({
            "feature": [last_col],
            "VIF": [float("nan")]
        })
        tables.append(final_vif)

    return tables, X_current

def save_vif_tables_to_excel_side_by_side(vif_tables, filename="vif_iterations.xlsx"):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    all_data = pd.DataFrame()
    max_len = max(len(t) for t in vif_tables)

    for i, table in enumerate(vif_tables):
        table_extended = table.reindex(range(max_len)).reset_index(drop=True)
        table_extended.columns = [f'feature_{i+1}', f'VIF_{i+1}']
        
        if i > 0:
            spacer = pd.DataFrame({f'space_{i}': [None]*max_len})
            all_data = pd.concat([all_data, spacer], axis=1)

        all_data = pd.concat([all_data, table_extended], axis=1)

    all_data.to_excel(writer, index=False, sheet_name='VIF_Steps')
    writer.close()
    print(f"✅ Saved to: {filename}")

vif_tables, X_filtered = iterative_vif(X)
save_vif_tables_to_excel_side_by_side(vif_tables)
