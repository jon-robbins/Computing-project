# data preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler


## impute NA
def impute_na(df, col_list):
    try:
        for item in col_list:
            df[item[0]] = df[item[0]].fillna(item[1])
        return df
    except:
        print("there's an error")
        return None


# delete columns with >40% nans or in irrelevant columns
def delete_unnec_cols(df, percentage=0.4, cols=['totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer']):
    try:
        col_check = []
        for col in df.columns:
            if df[col].isna().sum() > percentage * len(df) or col in cols:
                col_check.append(col)

        # Here we actually delete the columns
        for col in df.columns:
            if df[col].isna().sum() > percentage * len(df) or col in cols:
                del df[col]
        return df
    except:
        print("there's an error")
        return None


def to_num(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def standardize_data(df_comb, not_standardize_list):
    try:
        standardize_vars_cols = []
        dummy_vars_and_y = []
        for column in df_comb.columns:
            for element in not_standardize_list:

                if element in column:
                    save = False
                    break
                else:
                    save = True

            if save:
                standardize_vars_cols.append(column)
            else:
                dummy_vars_and_y.append(column)

        # scaler
        scaler = StandardScaler().fit(df_comb[standardize_vars_cols])
        scaled_features = scaler.transform(df_comb[standardize_vars_cols])
        df_scaled = pd.DataFrame(scaled_features, index=df_comb.index, columns=df_comb[standardize_vars_cols].columns)
        df_scaled[dummy_vars_and_y] = df_comb[dummy_vars_and_y]
        df_scaled.head()
        return df_scaled
    except:
        print("there's an error")
        return None
