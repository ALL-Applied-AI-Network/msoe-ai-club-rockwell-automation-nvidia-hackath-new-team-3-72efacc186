import unittest

from AI_Object_Perception.object_detection import (
    validate_annotations,
    summarize_objects,
)


class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.annotations = [
            {
                "object": "/World/SortVial0",
                "class": "tube",
                "bbox_xyxy": [100, 200, 300, 400],
                "occlusion_ratio": 0.0,
            },
            {
                "object": "/World/SortCap0",
                "class": "cap",
                "bbox_xyxy": [400, 500, 450, 550],
                "occlusion_ratio": 0.2,
            },
        ]

    def test_valid_annotations(self):
        self.assertTrue(validate_annotations(self.annotations))

    def test_missing_field(self):
        invalid = [self.annotations[0].copy()]
        del invalid[0]["class"]

        with self.assertRaises(ValueError):
            validate_annotations(invalid)

    def test_invalid_bbox(self):
        invalid = [self.annotations[0].copy()]
        invalid[0]["bbox_xyxy"] = [300, 200, 100, 400]

        with self.assertRaises(ValueError):
            validate_annotations(invalid)

    def test_invalid_occlusion(self):
        invalid = [self.annotations[0].copy()]
        invalid[0]["occlusion_ratio"] = 1.5

        with self.assertRaises(ValueError):
            validate_annotations(invalid)

    def test_class_counts(self):
        counts = summarize_objects(self.annotations)

        self.assertEqual(counts["tube"], 1)
        self.assertEqual(counts["cap"], 1)


if __name__ == "__main__":
    unittest.main()
