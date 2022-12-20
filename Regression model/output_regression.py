import pandas as pd 
from sklearn.linear_model import LinearRegression


def output_regression(df):
    regr = LinearRegression(fit_intercept=True)
    Y_pred_final = regr.predict(df)
    
    return Y_pred_final()

