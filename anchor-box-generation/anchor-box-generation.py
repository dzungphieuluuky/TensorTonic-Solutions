import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    res = []
    for i in range(feature_size):
        for j in range(feature_size):
            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    res.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return res