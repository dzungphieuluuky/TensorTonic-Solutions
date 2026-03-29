import numpy as np
def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    cat_to_idxs = {}
    for i in range(len(categories)):
        if categories[i] not in cat_to_idxs:
            cat_to_idxs[categories[i]] = [i]
        else:
            cat_to_idxs[categories[i]].append(i)
    mean_target = {}
    for key, indices in cat_to_idxs.items():
        mean_target[key] = np.sum([targets[index] for index in indices]) / len(indices)
    res = []
    for cate in categories:
        res.append(mean_target[cate])
    return res