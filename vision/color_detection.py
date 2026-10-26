from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data" / "tablecam_sample_003" / "rgb_0000.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

# Initial HSV thresholds for cube colors
COLOR_RANGES = {
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


def detect_colored_objects(image):
    """
    Detect colored objects from camera image pixels.

    Returns detections using Chris's class and bbox_xyxy fields.
    Does not use ground-truth annotations.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(hsv, lower, upper)
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Temporary ROI for Chris's sample TableCam
            if not (1500 <= x <= 2500 and 1100 <= y <= 1800):
                continue

            if area < 20 or area > 5000:
                continue

            # Experimental size filter for yellow cubes.
            # These limits are based on Chris's first sample image.
            if color_name == "cube_yellow":
                if w > 45 or h > 45:
                    continue

            detections.append({
                "class": color_name,
                "bbox_xyxy": [x, y, x + w, y + h]
            })

    return detections


def draw_detections(image, detections):
    """Draw predicted bounding boxes on a copy of the image."""
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            detection["class"],
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


if __name__ == "__main__":
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    detections = detect_colored_objects(image)

    for detection in detections:
        print(
            f"Detected: {detection['class']} "
            f"at {detection['bbox_xyxy']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "color_detections.png"

    annotated_image = draw_detections(image, detections)

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError("Could not save detection image")

    print(f"\nTotal detections: {len(detections)}")
    print(f"Saved output to: {output_path}")