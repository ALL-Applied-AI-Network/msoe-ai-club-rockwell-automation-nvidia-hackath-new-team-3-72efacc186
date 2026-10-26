# Object classes currently supported by our color detector
SUPPORTED_CLASSES = {
    "cube_red",
    "cube_blue",
    "cube_yellow"
}


def make_grasp_decision(detection):
    """
    Evaluate a detected object as a potential grasp target.

    This produces a planning decision, not a robot command.
    """

    object_class = detection["class"]
    x1, y1, x2, y2 = detection["bbox_xyxy"]

    # Calculate the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Check whether we currently support this object class
    if object_class in SUPPORTED_CLASSES:
        action = "candidate_for_grasp"
    else:
        action = "requires_review"

    return {
        "class": object_class,
        "action": action,
        "image_center": [center_x, center_y],
        "bbox_xyxy": [x1, y1, x2, y2]
    }


if __name__ == "__main__":

    # Example from our actual red cube detection
    sample_detection = {
        "class": "cube_red",
        "bbox_xyxy": [2410, 1437, 2445, 1472]
    }

    decision = make_grasp_decision(sample_detection)

    print("Grasp Decision Result:")
    print(f"Object: {decision['class']}")
    print(f"Action: {decision['action']}")
    print(f"Image Center: {decision['image_center']}")
    print(f"Bounding Box: {decision['bbox_xyxy']}")