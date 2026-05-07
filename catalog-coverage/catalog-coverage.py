def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    import numpy as np
    uniques = []
    for line in recommendations:
        for item in line:
            if item not in uniques:
                uniques.append(item)
    return len(uniques)/n_items