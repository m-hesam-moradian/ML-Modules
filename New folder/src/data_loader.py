import pandas as pd


def load_data(path):
    """
    Loads the Excel data
    """
    df = pd.read_excel(path)
    return df
