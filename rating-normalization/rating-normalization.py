def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    my_matrix = matrix.copy()
    for i in range(len(my_matrix)):
        sum = 0
        count = 0
        for j in range(len(my_matrix[i])):
            if my_matrix[i][j] != 0:
                sum += my_matrix[i][j]
                count += 1
        if count == 0:
            mean = 0
        else:
            mean = sum / count
        for j in range(len(my_matrix[i])):
            if my_matrix[i][j] != 0:
                my_matrix[i][j] -= mean
    return my_matrix