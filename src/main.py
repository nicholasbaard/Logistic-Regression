import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess
from logistic_regression import logistic_regression
from utils import plot_classification_report, plot_loss_curve

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/mushroom_cleaned.csv', help='Path to the dataset')
    parser.add_argument('--cols_to_drop', type=list, default=[], help='Columns to drop')
    parser.add_argument('--target_col', type=str, default='class', help='Target column')
    parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate for gradient descent')
    parser.add_argument('--eps', type=float, default=0.0001, help='Convergence threshold for gradient descent')
    parser.add_argument('--lambda', type=float, default=0.01, help='Regularization parameter')
    parser.add_argument('--scaler_type', type=str, default='minmax', help='Scaler type - minmax or standardization')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether you want the plots to be shown')
    args = vars(parser.parse_args())

    # Import dataset:
    df = pd.read_csv(args['dataset_path'])

    # preprocess the data
    X_train, X_test, y_train, y_test, theta, target_names = preprocess(df, 
                                                                        cols_to_drop=args['cols_to_drop'], 
                                                                        target_col=args['target_col'],
                                                                        scaler_type=args['scaler_type']
                                                                        )


    # Train the model
    if len(target_names) < 3:
        theta, cost_history = logistic_regression(X_train, y_train, theta, args['alpha'], args['eps'], args['lambda'])
    
    # Print the final weights and cost
    print(f"Final weights: {theta.ravel()}")
    print(f"Final cost: {cost_history[-1]}")

    # Print training MSE
    plot_classification_report(X_train, theta, y_train, target_names, train_test="train")

    # Plot the cost history
    plot_loss_curve(cost_history, show_plot=args['show_plot'])

    # Print test MSE
    plot_classification_report(X_test, theta, y_test, target_names, train_test="test")   

    ### Use sklearn for comparison ###

    # Create logistic regression object
    lr = LogisticRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_test)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)