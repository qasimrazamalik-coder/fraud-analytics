import unittest

import pandas as pd

from fraudlab import model


def frame(steps):
    return pd.DataFrame({"step": steps})


class Split(unittest.TestCase):
    def test_boundaries_with_delay_48(self):
        steps = [1, 52, 53, 99, 100, 152, 153, 199, 200, 300]
        fit, val, test, purged = model.temporal_split(frame(steps), val_start=100, test_start=200, delay=48)
        self.assertEqual(fit["step"].tolist(), [1, 52])  # 52 + 48 = 100 is the last usable fit row
        self.assertEqual(val["step"].tolist(), [100, 152])  # 152 + 48 = 200
        self.assertEqual(test["step"].tolist(), [200, 300])
        self.assertEqual(purged, 4)  # 53, 99, 153, 199

    def test_every_label_is_confirmed_before_the_next_block_starts(self):
        steps = list(range(1, 721))
        fit, val, test, _ = model.temporal_split(frame(steps), 360, 540, 48)
        self.assertLessEqual(fit["step"].max() + 48, val["step"].min())
        self.assertLessEqual(val["step"].max() + 48, test["step"].min())

    def test_blocks_are_disjoint_and_cover_all_but_purged(self):
        steps = list(range(1, 721))
        fit, val, test, purged = model.temporal_split(frame(steps), 360, 540, 48)
        self.assertEqual(len(fit) + len(val) + len(test) + purged, 720)
        self.assertEqual(len(set(fit.index) & set(val.index)) + len(set(val.index) & set(test.index)), 0)

    def test_rejects_bad_boundaries(self):
        with self.assertRaises(ValueError):
            model.temporal_split(frame([1, 2]), 200, 100)

    def test_split_points(self):
        self.assertEqual(model.split_points(frame([1, 720])), (360, 540))


if __name__ == "__main__":
    unittest.main()
