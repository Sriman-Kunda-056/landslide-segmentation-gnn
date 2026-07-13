"""Regression tests for portable image/mask discovery and binary masks."""

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import LandslideDataset


class DatasetTests(unittest.TestCase):
    def test_case_insensitive_scan_does_not_duplicate_files(self):
        temp_root = Path(__file__).resolve().parent / "tmp"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"dataset-test-{uuid.uuid4().hex}"
        try:
            images = root / "images"
            masks = root / "masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)

            Image.fromarray(
                np.zeros((8, 8, 3), dtype=np.uint8)
            ).save(images / "Tile01.PNG")
            Image.fromarray(
                np.ones((8, 8), dtype=np.uint8)
            ).save(masks / "tile01.png")

            dataset = LandslideDataset(
                str(images), str(masks), img_size=8, augment=False
            )
            self.assertEqual(len(dataset), 1)

            _, mask = dataset[0]
            self.assertEqual(set(mask.unique().tolist()), {1})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
