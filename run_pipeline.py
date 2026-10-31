import json
from pathlib import Path

import cv2

from AI_Object_Perception.color_detection import detect_colored_objects
from AI_Grasp_Candidate_Selection.grasp_decision import make_grasp_decision


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "tablecam_sample_003"
    / "rgb_0000.png"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "grasp_decisions.json"


def run_pipeline(image_path):
    """
    Run object detection and preliminary grasp reasoning.

    Input: Camera image
    Output: List of grasp decisions
    """

    # Step 1: Load the camera image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load camera image: {image_path}"
        )

    # Step 2: Detect objects using image pixels
    detections = detect_colored_objects(image)

    # Step 3: Generate a decision for each detected object
    decisions = []

    for detection in detections:
        decision = make_grasp_decision(detection)
        decisions.append(decision)

    return decisions


if __name__ == "__main__":

    # Run our vision + reasoning pipeline
    decisions = run_pipeline(IMAGE_PATH)

    # Create the output folder if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save decisions in JSON format
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(decisions, file, indent=2)

    # Display results in the terminal
    print("\n=== Vessels of Cosmos: AI Pipeline ===")

    print(f"Total detections: {len(decisions)}")

    for index, decision in enumerate(decisions, start=1):
        print(
            f"{index}. {decision['class']} | "
            f"Action: {decision['action']} | "
            f"Center: {decision['image_center']}"
        )

    print(f"\nDecisions saved to: {OUTPUT_PATH}")