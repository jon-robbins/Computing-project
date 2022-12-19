# data preprocessing
import pandas as pd


## impute NA
def impute_na(df, col_list):
    try:
        for item in col_list:
            print(item[0])
            df[item[0]] = df[item[0]].fillna(item[1])

        return df
    except:
        print("发生异常")
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
        print("发生异常")
        return None


def to_num(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
