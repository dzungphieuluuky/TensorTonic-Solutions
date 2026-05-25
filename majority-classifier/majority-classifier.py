import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    uniques, counts = np.unique(y_train, return_counts=True)
    freq_idx = np.argmax(counts)
    freq_val = uniques[freq_idx]
    return np.full(X_test.shape, freq_val)