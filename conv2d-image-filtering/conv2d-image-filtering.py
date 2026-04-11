import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution (cross-correlation) to a single-channel image.
    
    Args:
        image: 2D numpy array or list of lists
        kernel: 2D numpy array or list of lists
        stride: int, step size for sliding window
        padding: int, number of zero rows/cols added to each side
    
    Returns:
        2D numpy array of convolved result
    """
    image = np.array(image)
    kernel = np.array(kernel)
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    
    # Output dimensions
    out_h = (img_h + 2 * padding - kernel_h) // stride + 1
    out_w = (img_w + 2 * padding - kernel_w) // stride + 1
    
    # Pad the image
    padded = np.zeros((img_h + 2 * padding, img_w + 2 * padding))
    padded[padding:padding+img_h, padding:padding+img_w] = image
    
    # Prepare output
    output = np.zeros((out_h, out_w))
    
    # Perform convolution (cross-correlation)
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region of interest
            region = padded[i*stride : i*stride + kernel_h,
                            j*stride : j*stride + kernel_w]
            output[i, j] = np.sum(region * kernel)
    
    return output.tolist()