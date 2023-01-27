import torch
import unittest

from .utils.data import DummyDataset
from .utils.scoring import content_area_hausdorff, MISS_THRESHOLD

from torchcontentarea.utils import draw_area, crop_area

TESTS_CASES = ["cuda", "cpu"]

class TestUtils(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = DummyDataset()

    def test_draw_area(self):
        _, mask, area = self.dataset[0]
        for device in TESTS_CASES:
            with self.subTest(device):
                mask = mask.to(device)
                result = draw_area(area, mask)
                score = torch.where(1 - result == mask, 0, 1).sum()
                self.assertLess(score, 1)
                
    def test_crop_area(self):
        _, mask, area = self.dataset[0]
        for device in TESTS_CASES:
            with self.subTest(device):
                mask = mask.to(device)
                result = crop_area(area, mask)
                self.assertTrue(result.unique().numel() == 1)
                
if __name__ == '__main__':
    unittest.main()