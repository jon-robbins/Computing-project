import pandas as pd
import numpy as np



# MAE
def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff


# MSE
def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   
   return mean_diff

# R-squared
def rsquared(act, pred):
    
    y_bar = act.mean()
    ss_tot = ((act - y_bar)**2).sum()
    ss_res = ((act - pred)**2).sum()
    return 1 - (ss_res / ss_tot)