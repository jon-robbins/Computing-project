# feature creation
import pandas as pd


def create_dummies(df, cols):
    try:
        df = pd.get_dummies(df, columns=cols)
        return df
    except:
        print("there's an error")
        return None


def upper_outlier_dummy(df, col: str):
    from scipy.stats import iqr
    import numpy as np
    iqr = iqr(df.col, nan_policy='omit') * 1.5

    q3 = np.nanquantile(df.col, 0.75)
    upper_bound = iqr + q3

    df.loc[df.col > upper_bound, f'{col}_upper_outlier'] = 1
    df.loc[df.col <= upper_bound, f'{col}upper_outlier'] = 0
    return df


def lower_outlier_dummy(df, col: str):
    from scipy.stats import iqr
    import numpy as np
    iqr = iqr(df.col, nan_policy='omit') * 1.5

    q1 = np.nanquantile(df.col, 0.25)
    lower_bound = q1 - iqr

    df.loc[df.col < lower_bound, f'{col}_upper_outlier'] = 1
    df.loc[df.col >= lower_bound, f'{col}upper_outlier'] = 0
    return df
