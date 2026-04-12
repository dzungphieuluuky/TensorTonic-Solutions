def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    import numpy as np
    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)
    ref_prob = reference_counts / np.sum(reference_counts)
    prod_prob = production_counts / np.sum(production_counts)
    score = (np.sum(np.abs(ref_prob - prod_prob)))/2
    if score > threshold:
        return {
            "score": score,
            "drift_detected": True
        }
    else:
        return {
            "score": score,
            "drift_detected": False
        }
        