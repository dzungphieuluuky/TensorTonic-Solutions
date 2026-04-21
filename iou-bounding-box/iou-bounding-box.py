def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    x1, y1, x2, y2 = box_a
    new_x1, new_y1, new_x2, new_y2 = box_b
    intersect = [max(x1, new_x1), max(y1, new_y1), min(x2, new_x2), min(y2, new_y2)]
    if intersect[2] < intersect[0] and intersect[3] < intersect[1]:
        intersect_area = 0
    else:
        intersect_area = (intersect[2] - intersect[0]) * (intersect[3] - intersect[1])
    
    box1 = (x2 - x1) * (y2 - y1)
    box2 = (new_x2 - new_x1) * (new_y2 - new_y1)
    union = box1 + box2 - intersect_area
    return intersect_area/union