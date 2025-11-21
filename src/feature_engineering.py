import pandas as pd
import numpy as np


def convert_to_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts latitude and longitude (in degrees) to 3D Cartesian coordinates (X, Y, Z).

    The formulas are:
    X = cos(lat) * cos(lon)
    Y = cos(lat) * sin(lon)
    Z = sin(lat)

    Args:
        df: Input DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
        DataFrame with three new columns: 'X', 'Y', and 'Z'.
    """
    df = df.copy()

    # Convert degrees to radians for trigonometric functions
    lat_rad = np.radians(df['latitude'])
    lon_rad = np.radians(df['longitude'])

    # Apply conversion formulas
    df['X'] = np.cos(lat_rad) * np.cos(lon_rad)
    df['Y'] = np.cos(lat_rad) * np.sin(lon_rad)
    df['Z'] = np.sin(lat_rad)

    return df

def apply_log1p_transformation(df: pd.DataFrame,
                               target_cols: list,
                               drop_original: bool = False) -> pd.DataFrame:
    df = df.copy()

    original_cols_to_drop = []

    for col in target_cols:
        if col not in df.columns:
            print(f"'{col}' not in DataFrame, continue。")
            continue

        new_col = f'{col}_log'

        data_to_transform = df[col]

        # check：log(1+x) -> x >= -1
        if (data_to_transform < 0).any():
            print(f"'{col}' contain negative value")

        # log(1 + x)
        df[new_col] = np.log1p(data_to_transform)

        if drop_original:
            original_cols_to_drop.append(col)

    # true -> drop original columns
    if drop_original and original_cols_to_drop:
        df = df.drop(columns=original_cols_to_drop)

    return df

def apply_one_hot_encoding(df: pd.DataFrame,
                           target_cols: list,
                           drop_first: bool = True) -> pd.DataFrame:

    df = df.copy()

    df_encoded = pd.get_dummies(
        df,
        columns=target_cols,
        prefix=target_cols,
        drop_first=drop_first
    )

    return df_encoded


