def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    import numpy as np
    X = np.asarray(X)
    y = np.asarray(y)
    N, D = X.shape
    pre_inverse = (X.T @ X + lam * np.eye(D))
    result = np.linalg.inv(pre_inverse) @ X.T @ y
    return result