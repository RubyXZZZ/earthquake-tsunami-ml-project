import pandas as pd
import numpy as np


def impute_with_median(df: pd.DataFrame,
                       target_cols: list) -> pd.DataFrame:
    """
    impute with NaN with median。
    """

    df = df.copy()

    for col in target_cols:
        if col not in df.columns:
            print(f"'{col}' not in DataFrame")
            continue

        # impute with median
        median_val = df[col].median()

        df[col] = df[col].fillna(median_val)

    return df


def impute_with_conditional_median(df: pd.DataFrame,
                                   target_col: str,
                                   condition_col: str = 'magnitude',
                                   bin_size: float = 0.5):

    df = df.copy()

    # create bin
    bin_col = f"{condition_col}_bin"
    df[bin_col] = (df[condition_col] / bin_size).round().astype(int)

    # median table
    median_map = df.groupby(bin_col)[target_col].median()

    # mapped values
    mapped = df[bin_col].map(median_map)

    # mask
    mask = df[target_col].isna()

    # --- KEY FIX: direct assignment, no fillna(series) ---
    df.loc[mask, target_col] = mapped[mask].to_numpy()

    # fallback global median
    global_med = df[target_col].median()
    df.loc[df[target_col].isna(), target_col] = global_med

    # remove temp column
    df.drop(columns=[bin_col], inplace=True)

    return df



