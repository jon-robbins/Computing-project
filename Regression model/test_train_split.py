#Create training and test split

#Note, in class we were given both training and test data sets. For this, I'm assuming we're getting a training set, then splitting it into test and training, then remove the target variable for the test set

from sklearn.model_selection import train_test_split

def test_train_split(df, dropna=bool):
    if dropna == True:
        df = df.dropna()
    else:
        pass
    X = df.drop(columns = ['roomnum', 'ventura_county', 'lotid','heatingtype', 'numbath', 'heatingdesc', 'countycode', 'citycode', 'countycode2', 'taxyear', 'parcelvalue', 'neighborhoodcode', 'regioncode', 'numfireplace', 'unitnum', 'hd1', 'hd2', 'hd3', 'hd4', 'organge_county'], axis = 1)

    y = df['parcelvalue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 43)
    return X_train, X_test, y_train, y_test
