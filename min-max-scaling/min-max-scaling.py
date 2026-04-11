def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    import numpy as np
    data = np.array(data)
    maximum = np.max(data, axis=0, keepdims=True)
    minimum = np.min(data, axis=0, keepdims=True)
    new_data = np.where(maximum - minimum == 0, 0, (data - minimum)/(maximum - minimum))
    return new_data.tolist()