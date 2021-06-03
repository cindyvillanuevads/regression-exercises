import pandas as pd
import numpy as np
import os
import acquire 

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# ****************************************************************************************************************************************
#  this function was shared by a classmate
#*******************************************************************************************************************************\

def missing_values_table(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values, and the percent of that column that has missing values
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           "columns that have missing values.")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns







# *******************************************************************************************************


def clean_telco(df):
    ''''
    This function will get customer_id, monthly_charges, tenure, and total_charges 
    from the previously acquired telco df, for all customers with a 2-year contract.
    drop any duplicate observations, 
    conver total_charges to a float type.
    return cleaned telco DataFrame
    '''
    #getting only the customers who have 2 year contract using the condition df.contract_type_id == 3
    telco_df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']][df.contract_type_id == 3]
    #drop duplicates
    telco_df = telco_df.drop_duplicates()
    # add a '0' only to the columns that have " "
    telco_df[telco_df['total_charges']== ' '] = telco_df[telco_df['total_charges']== ' '].replace(' ','0')
    # convert total_charges to float
    telco_df['total_charges']= telco_df['total_charges'].astype('float')
        
    return telco_df



def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test


def wrangle_telco():
    ''''
    This function will acquire telco db using get_telco function. then it will use another
    function named  clean_telco that create a new df only with  customer_id, monthly_charges, tenure, and total_charges 
    from the previously acquired telco df, this new df will contain only customers with a 2-year contract.
    drop any duplicate observations, 
    conver total_charges to a float type.
    return cleaned telco DataFrame
    '''
    df = acquire.get_telco()
    telco_df = clean_telco(df)
    return telco_df
    