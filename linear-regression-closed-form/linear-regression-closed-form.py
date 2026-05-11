import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    XTX = X.T @ X
    weights = np.linalg.inv(XTX) @ X.T @ y
    return weights