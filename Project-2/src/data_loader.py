import pandas as pd


def load_data(path):
    """
    Loads the Excel data and removes the time column.
    Returns a DataFrame.
    """
    df = pd.read_excel(path)

    # Drop the timestamp
    if "Timestamp" in df.columns:
        df.drop(columns=["Timestamp"], inplace=True)

    return df
