
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from math import sqrt

def plot_residuals ( df, y, yhat  ):
    '''
    Takes in a dataframe , y = column with actual_values and yhat= name of the columns with predicted_values
    and creates a residual plot
    
    Example:
    plot_residuals(df, 'tip', 'yhat')
    '''

    #baseline
    df['yhat_baseline'] = df[y].mean()


    # residuals
    df['residuals'] = df[yhat] - df[y]
    df['residuals_baseline'] = df ['yhat_baseline'] - df[y]

    # plot
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.hist(df.residuals_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residuals, label='model residuals', alpha=.6)
    ax.legend()
    plt.show()
    return




def defregression_errors(df, y, yhat):
    '''
    Takes in a dataframe , y = column with actual_values and yhat= name of the columns with predicted_values
    and calculate:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    
    Example:
    plot_residuals(df, 'tip', 'yhat')
    '''
    #import
    from sklearn.metrics import  mean_squared_error
    from math import sqrt
    
    
    #calculate SSE using sklearn
    SSE = mean_squared_error(df[y], df[yhat])*len(df)
    #explained sum of squares (ESS)
    ESS = ((df[yhat] - df[y].mean())**2).sum()
    #total sum of squares (TSS)
    TSS = ((df[y] - df[y].mean())**2).sum()
    #mean squared error (MSE)
    MSE = mean_squared_error(df[y], df[yhat])
    #root mean squared error (RMSE)
    RMSE = sqrt(MSE)
    
    #print
    print ('**MODEL**')
    print (f' Sum of squared errors (SSE)    = {round (SSE, 3)}')
    print (f' Explained sum of squares (ESS) = {round (ESS, 3)}')
    print (f' Total sum of squares (TSS)     = {round (TSS, 3)}')
    print (f' mean squared error (MSE)       = {round (MSE, 3)}')
    print (f' mean squared error (RMSE)      = {round (RMSE, 3)}')
    print (f' Variance (r2)                  = {round (ESS/TSS, 3)}')
    print ('  ')
    return {
        'sse': SSE,
        'ess': ESS,
        'tss': TSS,
        'mse': MSE,
        'rmse': RMSE,
        'r2': ESS/TSS,
    }



def baseline_mean_errors(df, y):
    '''
    Takes in a dataframe , y = column with actual_values 
    and calculate:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)

    Example:
    plot_residuals(df, 'tip')
    '''
    #import
    from sklearn.metrics import  mean_squared_error
    from math import sqrt

    #baseline
    df['yhat_baseline'] = df[y].mean()

    #calculate SSE using sklearn
    SSE_baseline = mean_squared_error(df[y], df['yhat_baseline'])*len(df)
    #mean squared error (MSE)
    MSE_baseline = mean_squared_error(df[y], df.yhat_baseline)
    #root mean squared error (RMSE)
    RMSE_baseline = sqrt(MSE_baseline)

    #print
    print ('**BASELINE**')
    print (f' Sum of squared errors (SSE)    = {round (SSE_baseline, 3)}')
    print (f' mean squared error (MSE)       = {round (MSE_baseline, 3)}')
    print (f' mean squared error (RMSE)      = {round (RMSE_baseline, 3)}')
    print ('  ')
    return {
        'sse': SSE_baseline,
        'mse': MSE_baseline,
        'rmse': RMSE_baseline,
    }