import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scale_data(X_train:np.array, 
               X_test:np.array, 
               y_train:np.array, 
               y_test:np.array, 
               scaler_type:str='minmax'
               ):
    """
    Scale the data using either min-max scaling or standardization.
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    # y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def map_target_column(df, target_col):
    target_names = df[target_col].unique()
    if df[target_col].dtype == object: 
        df[target_col] = df[target_col].map({v: k for k, v in enumerate(target_names, start=0)})

    return df, target_names


def split_data(df:pd.DataFrame, target_column:str):
    """
    Split the data into training and testing sets.
    """

    X = df.drop([target_column], axis=1)
    y = df[target_column]
    y = np.reshape(y, (len(y), 1))

    # Add a column of ones to X for the bias term
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Initialize weights to zero
    theta = np.ones((X.shape[1], 1))

    return X_train, X_test, y_train, y_test, theta


def preprocess(df:pd.DataFrame, 
               cols_to_drop:list[str], 
               target_col:str, 
               scaler_type='minmax'
               ):
    """
    Preprocess the data for linear regression.
    """

    # drop date and store number columns
    df = df.drop(cols_to_drop, axis=1)

    # map text target column
    df, target_names, = map_target_column(df, target_col)

    # Split the data
    X_train, X_test, y_train, y_test, theta = split_data(df, target_column=target_col)

    # Scale the data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = scale_data(X_train, X_test, y_train, y_test, scaler_type)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, theta, target_names

