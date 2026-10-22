def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.

    Format: [x1, y1, x2, y2]
    """

    # Find the overlapping rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate overlapping area
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    # Calculate area of each bounding box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate combined area
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


if __name__ == "__main__":

    # Chris's actual red cube annotation
    ground_truth = [2410, 1436, 2445, 1472]

    # Our OpenCV prediction
    prediction = [2410, 1437, 2445, 1472]

    iou = calculate_iou(ground_truth, prediction)

    print(f"Ground truth: {ground_truth}")
    print(f"Prediction:   {prediction}")
    print(f"IoU Score:    {iou:.4f}")
    print(f"IoU Percent:  {iou * 100:.2f}%")