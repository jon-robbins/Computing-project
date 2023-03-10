# Create training and test split

from sklearn.model_selection import train_test_split


def test_train_split(df, target_col, test_size=0.2, random_state=43, dropna=bool):
    if dropna:
        df = df.dropna()
    else:
        pass
    X = df.drop(columns=[target_col], axis=1)

    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
