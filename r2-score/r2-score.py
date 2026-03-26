import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.sum((y_true - y_pred)**2)
    deno = np.sum((y_true - np.mean(y_true))**2)
    if deno == 0:
        if np.sum(np.abs(y_true - y_pred)) == 0:
            return 1.0
        else:
            return 0.0
    R2 = 1 - (numerator/deno)
    return R2