def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    result = []
    n, m = len(image), len(image[0])
    for i in range(n):
        temp = []
        for j in range(m):
            red = image[i][j][0]
            green = image[i][j][1]
            blue = image[i][j][2]
            gray = 0.299 * red + 0.587 * green + 0.114 * blue
            temp.append(gray)
        result.append(temp)
    return result