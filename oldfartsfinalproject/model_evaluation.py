import pandas as pd
import numpy as np



# MAE
def mean_absolute_error(act, pred):
    """
    Agrs: (act,pred)

    Calculate mean absolute error by comparing the actual test data output and the model predicted ouptut.

    Returns the value of MAE
    """
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff


# MSE
def mean_squared_error(act, pred):
    """
    Agrs: (act,pred)

    Calculate mean squared error by comparing the actual test data output and the model predicted ouptut.

    Returns the value of MAE
    """
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()

    return mean_diff

# R-squared
def rsquared(act, pred):
    """
    Agrs: (act,pred)

    Calculate the value of R squared by comparing the actual test data output and the model predicted ouptut.

    Returns the value of MAE
    """
    y_bar = act.mean()
    ss_tot = ((act - y_bar)**2).sum()
    ss_res = ((act - pred)**2).sum()
    return 1 - (ss_res / ss_tot)