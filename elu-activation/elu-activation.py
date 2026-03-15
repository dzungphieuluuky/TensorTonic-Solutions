import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [ele if ele > 0 else alpha * (math.exp(ele) - 1) for ele in x]
    