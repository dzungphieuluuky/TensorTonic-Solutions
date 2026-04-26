import numpy as np

# need to review this problem once more because of sickness at the time submitting this solution
def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    n_samples = X.shape[0]
    # Step 1: Compute the Euclidean distance matrix (n_samples x n_samples)
    # Using broadcasting: (n,1,d) - (1,n,d) -> (n,n,d) then sum of squares
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=-1))   # D[i,j] = distance between point i and j
    
    # Step 2: Pre‑compute which points belong to each cluster
    unique_labels = np.unique(labels)
    cluster_masks = {lab: (labels == lab) for lab in unique_labels}
    
    silhouette_values = []
    
    for i in range(n_samples):
        c_i = labels[i]                     # cluster of point i
        same_mask = cluster_masks[c_i].copy()
        same_mask[i] = False                # exclude the point itself
        indices_same = np.where(same_mask)[0]
        
        # a(i) = mean distance to other points in the same cluster
        if len(indices_same) == 0:
            # Singleton cluster: silhouette is defined as 0
            a_i = 0.0
            # For b(i) we still need to consider other clusters.
            # If a_i == 0, the formula gives 1 when b_i > 0,
            # but standard practice (e.g. sklearn) sets s_i = 0 for singletons.
            s_i = 0.0
        else:
            a_i = np.mean(D[i, indices_same])
            
            # b(i) = smallest mean distance to points in any *other* cluster
            b_i = np.inf
            for lab, mask in cluster_masks.items():
                if lab == c_i:
                    continue
                indices_other = np.where(mask)[0]
                if len(indices_other) > 0:
                    mean_dist = np.mean(D[i, indices_other])
                    if mean_dist < b_i:
                        b_i = mean_dist
            
            # b_i must be defined (at least one other cluster exists)
            if np.isinf(b_i):
                # Only one cluster overall – silhouette undefined.
                # Returning a default of 0 (or could raise an error).
                s_i = 0.0
            else:
                # Standard formula
                s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_values.append(s_i)
    
    # Step 3: Return the mean over all points
    return np.mean(silhouette_values)