import pandas as pd
import numpy as np
import os
import acquire 

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# ****************************************************************************************************************************************
#  this function was shared by a classmate. I add some things
#*******************************************************************************************************************************\

def miss_dup_values(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows, 
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
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
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns







# ********************************************  TELCO   ***********************************************************


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

def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test

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



# ******************************************** ZILLOW ********************************************

def clean_zillow (df):
    '''
    Takes in a df and drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800, and bedroomcnt, yearbuilt, fips are changed to
    int.
    Return a clean df
    '''
    
    # drop duplicates
    df = df.drop_duplicates()
    #drop nulls
    df = df.dropna(how='any',axis=0)

    #drop all houses with bath = 0 and bedromms = 0
    #get the index to drop the rows
    ind = list(df[(df.bedroomcnt == 0) & (df.bathroomcnt == 0)].index)
    #drop
    df.drop(ind, axis=0, inplace= True)


    #drop all houses calculatedfinisheedsqf <800
    #get the index to drop
    lis =list(df[df['calculatedfinishedsquarefeet'] < 800].index)
    #drop the rows
    df.drop(lis, axis=0, inplace = True)

    #bedrooms, yearbuilt and fips can be converted to int
    df[['bedroomcnt', 'yearbuilt', 'fips']] = df[['bedroomcnt', 'yearbuilt', 'fips']].astype(int)
    return df



def wrangle_zillow():
    ''''
    This function will acquire zillow db using get_new_zillow function. then it will use another
    function named  clean_zillwo that drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800.
     bedroomcnt, yearbuilt, fips are changed to int.
    return cleaned zillow DataFrame
    '''
    df = acquire.get_new_zillow()
    zillow_df = clean_zillow(df)
    return zillow_df




# Function for acquiring and prepping my student_grades df.

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')
    
    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = grades.dropna()
    
    # Convert all columns to int64 data types.
    df = df.astype('int')
    
    return df






def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(
        df, "G3"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )