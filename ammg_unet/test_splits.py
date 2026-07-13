"""Regression tests for the four-condition fine-tuning split."""

import unittest

from dataset import make_split_indices


class SplitIndexTests(unittest.TestCase):
    def test_lengths_and_disjointness(self):
        train, val, test = make_split_indices(100, 0.6, seed=42)

        self.assertEqual(len(train), 60)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(test), 20)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))
        self.assertEqual(len(set(train + val + test)), 100)

    def test_small_training_set_is_nested_and_eval_sets_are_fixed(self):
        train20, val20, test20 = make_split_indices(100, 0.2, seed=42)
        train60, val60, test60 = make_split_indices(100, 0.6, seed=42)

        self.assertEqual(len(train20), 20)
        self.assertTrue(set(train20).issubset(train60))
        self.assertEqual(val20, val60)
        self.assertEqual(test20, test60)

    def test_split_is_deterministic(self):
        first = make_split_indices(101, 0.2, seed=7)
        second = make_split_indices(101, 0.2, seed=7)
        different_seed = make_split_indices(101, 0.2, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)

    def test_invalid_training_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            make_split_indices(100, 0.0)
        with self.assertRaises(ValueError):
            make_split_indices(100, 0.7)


if __name__ == "__main__":
    unittest.main()
