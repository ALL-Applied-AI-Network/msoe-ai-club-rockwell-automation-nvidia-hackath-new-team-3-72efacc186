from pathlib import Path
import cv2

from vision.color_detection import detect_colored_objects
from vision.object_detection import load_annotations



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
    project_root = Path(__file__).resolve().parent.parent

    image_path = project_root / "data" / "tablecam_sample_003" / "rgb_0000.png"
    labels_path = project_root / "data" / "tablecam_sample_003" / "annotations_readable.json"

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    predictions = detect_colored_objects(image)
    annotations = load_annotations(labels_path)

    # Evaluate only cube classes for this baseline.
    cube_classes = {"cube_red", "cube_blue", "cube_yellow"}

    ground_truth = [
        obj for obj in annotations
        if obj["class"] in cube_classes
    ]

    matched = set()
    true_positives = 0
    false_positives = 0
    iou_scores = []

    for prediction in predictions:
        best_iou = 0.0
        best_index = None

        for index, actual in enumerate(ground_truth):
            if index in matched:
                continue

            if prediction["class"] != actual["class"]:
                continue

            iou = calculate_iou(
                prediction["bbox_xyxy"],
                actual["bbox_xyxy"]
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= 0.5:
            matched.add(best_index)
            true_positives += 1
            iou_scores.append(best_iou)
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )

    average_iou = (
        sum(iou_scores) / len(iou_scores)
        if iou_scores else 0.0
    )

    print(f"Ground-truth cubes: {len(ground_truth)}")
    print(f"Predictions: {len(predictions)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Average matched IoU: {average_iou:.2%}")