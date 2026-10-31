import json
from pathlib import Path

import cv2

# Find the project root based on this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chris's synthetic dataset
DATA_DIR = PROJECT_ROOT / "data" / "tablecam_sample_003"
IMAGE_PATH = DATA_DIR / "rgb_0000.png"
LABELS_PATH = DATA_DIR / "annotations_readable.json"

# Keep generated results separate from the original dataset.
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "detected_objects.png"

# Read the image.
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Read the JSON labels.
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    annotations = json.load(file)

print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
print(f"Total objects: {len(annotations)}")

# Draw a bounding box and label for each object.
for annotation in annotations:
    object_class = annotation["class"]
    x1, y1, x2, y2 = map(int, annotation["bbox_xyxy"])

    # Draw a green rectangle around the object.
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Write the object's class name.
    cv2.putText(
        image,
        object_class,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    print(f"Found: {object_class} at ({x1}, {y1}, {x2}, {y2})")

# Save the annotated image.
if not cv2.imwrite(str(OUTPUT_PATH), image):
    raise RuntimeError("Failed to save annotated image")

print(f"\nSuccess! Saved result to: {OUTPUT_PATH}")


