def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    current_x = x0
    for i in range(steps):
        current_derivative = 2 * a * current_x + b
        current_x = current_x - lr * current_derivative
    return current_x