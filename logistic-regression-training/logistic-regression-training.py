import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.

    Args:
        X: Input features of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        lr: Learning rate
        steps: Number of gradient descent iterations

    Returns:
        (w, b):
            w -> weight vector of shape (n_features,)
            b -> scalar bias
    """

    # Number of samples and features
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0

    # Gradient descent loop
    for step in range(steps):

        # Linear model
        z = X @ w + b

        # Predicted probabilities
        y_pred = _sigmoid(z)

        # Compute gradients
        dw = (1 / n_samples) * (X.T @ (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b