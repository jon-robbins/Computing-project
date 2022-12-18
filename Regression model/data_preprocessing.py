#data preprocessing

import pandas as pd

# Import the data
training = pd.read_csv('Regression_Supervised_Train.csv', sep = ";")
test = pd.read_csv('Regression_Supervised_Test.csv', sep = ";")

## impute NA
def impute_na(df):
    df['parcelvalue'] = pd.to_numeric(df['parcelvalue'], errors='coerce')

    df['poolnum'] = df['poolnum'].fillna(0)

    df['unitnum'] = df['unitnum'].fillna(1)
    # Turn Aircond into a dummy, the 4th column in the data
    df['aircond'] = df['aircond'].fillna(0)
    return df

#delete columns with >40% nans or in irrelevant columns
def delete_unnec_cols(df):
    col_check = []
    for col in df.columns:
        if df[col].isna().sum() > 0.4 * len(df) or col in ('totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer'):
            col_check.append(col)
            
    # Here we actually delete the columns        
    for col in df.columns:
        if df[col].isna().sum() > 0.4 * len(df) or col in ('totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer'):
            del df[col]
    return df
