import pandas as pd


def average_daily(df, samples_per_day=96):
    """
    Aggregates every 96 samples (1 day) to 1 row by averaging.
    Assumes rows are ordered by time.
    """
    n_days = len(df) // samples_per_day
    df_trimmed = df.iloc[: n_days * samples_per_day]  # Trim extra rows if not divisible
    daily_avg = df_trimmed.groupby(df_trimmed.index // samples_per_day).mean()
    return daily_avg
