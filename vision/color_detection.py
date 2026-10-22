from pathlib import Path
import cv2
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data" / "tablecam_sample_003" / "rgb_0000.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Chris's camera image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

# Convert image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Initial HSV ranges for our three cube colors
# These are starting thresholds, not yet calibrated.
color_ranges = {
    "cube_red": [
        (np.array([0, 100, 80]), np.array([10, 255, 255])),
        (np.array([170, 100, 80]), np.array([179, 255, 255]))
    ],
    "cube_blue": [
        (np.array([95, 100, 70]), np.array([135, 255, 255]))
    ],
    "cube_yellow": [
        (np.array([18, 100, 100]), np.array([38, 255, 255]))
    ]
}

detections = []

for color_name, ranges in color_ranges.items():
    # Create a mask for this color
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, lower, upper)
        )

    # Find colored regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        # Get the object's bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Only consider objects within the tabletop region.
        # Initial bounds based on Chris's sample image.
        if not (1500 <= x <= 2500 and 1100 <= y <= 1800):
            continue

        # Ignore tiny regions and large background areas
        if area < 20 or area > 5000:
            continue

        detection = {
            "class": color_name,
            "bbox_xyxy": [x, y, x + w, y + h]
        }

        detections.append(detection)

        # Draw bounding box and label
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            color_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(f"Detected: {color_name} at {detection['bbox_xyxy']}")

# Save annotated image
output_path = OUTPUT_DIR / "color_detections.png"

if not cv2.imwrite(str(output_path), image):
    raise RuntimeError("Could not save detection image")

print(f"\nTotal detections: {len(detections)}")
print(f"Saved output to: {output_path}")