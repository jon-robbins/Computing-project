#Create training and test split

#Note, in class we were given both training and test data sets. For this, I'm assuming we're getting a training set, then splitting it into test and training, then remove the target variable for the test set

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
training = pd.read_csv('Regression_Supervised_Train.csv', sep = ";")

# def split_test_train(df):
    

tr = training.copy()
tr = tr.dropna()
fyv = np.percentile(tr['parcelvalue'], 5)
nty = np.percentile(tr['parcelvalue'], 90)

tr = tr[(tr['parcelvalue'] < 1000000) & (tr['parcelvalue'] > fyv)]

# We check the "candidate" data one last time
# profile2 = ProfileReport(tr, title="Report")
# profile2

# potential is just an array with meaning full regressor and makes trying out models easier (Ctrl+C & Ctrl+V) 
potential = ['latitude', 'longitude', 'numbath', 'ventura_county']
X = tr.drop(columns = ['roomnum', 'ventura_county', 'lotid','heatingtype', 'numbath', 'heatingdesc', 'countycode', 'citycode', 'countycode2', 'taxyear', 'parcelvalue', 'neighborhoodcode', 'regioncode', 'numfireplace', 'unitnum', 'hd1', 'hd2', 'hd3', 'hd4', 'organge_county'], axis = 1)

y = tr['parcelvalue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 43)
