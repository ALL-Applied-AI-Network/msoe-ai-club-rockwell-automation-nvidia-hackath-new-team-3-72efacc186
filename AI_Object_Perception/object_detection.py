import json
from pathlib import Path


def load_annotations(json_path):
    """Load object annotations from Chris's JSON format."""
    with open(json_path, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, list):
        raise ValueError("Annotations must be a JSON list.")

    return annotations


def validate_annotations(annotations):
    """Check whether each annotation contains valid object data."""
    required_fields = ["object", "class", "bbox_xyxy", "occlusion_ratio"]

    for index, annotation in enumerate(annotations):
        for field in required_fields:
            if field not in annotation:
                raise ValueError(
                    f"Object {index} is missing required field: {field}"
                )

        bbox = annotation["bbox_xyxy"]

        if len(bbox) != 4:
            raise ValueError(f"Object {index} has an invalid bounding box.")

        x1, y1, x2, y2 = bbox

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Object {index} has incorrect bbox coordinates.")

        occlusion = annotation["occlusion_ratio"]

        if not 0 <= occlusion <= 1:
            raise ValueError(f"Object {index} has invalid occlusion ratio.")

    return True


def summarize_objects(annotations):
    """Count how many objects belong to each class."""
    counts = {}

    for annotation in annotations:
        object_class = annotation["class"]
        counts[object_class] = counts.get(object_class, 0) + 1

    return counts


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    json_path = (
        project_root
        / "data"
        / "tablecam_sample_003"
        / "annotations_readable.json"
    )

    annotations = load_annotations(json_path)

    validate_annotations(annotations)

    print("Annotations validated successfully!")
    print(f"Total objects: {len(annotations)}")

    for object_class, count in summarize_objects(annotations).items():
        print(f"{object_class}: {count}")

        