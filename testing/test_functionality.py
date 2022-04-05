import unittest
import torch

from utils import DummyDataset, TestDataLoader, iou_score
from torchcontentarea import ContentAreaInference

class TestAPI(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataloader = TestDataLoader(DummyDataset())
        self.content_area_inference = ContentAreaInference()

    def test_infer_area(self):
        for img, mask, area in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            infered_area = self.content_area_inference.infer_area(img)
            for x, y in zip(area, infered_area):
                self.assertAlmostEqual(x, y, delta=2)

    def test_infer_mask(self):
        for img, mask, _ in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            infered_mask = self.content_area_inference.infer_mask(img)
            score = iou_score(mask, infered_mask)
            self.assertGreater(score, 0.99)

    def test_draw_mask(self):
        for img, mask, area in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            drawn_mask = self.content_area_inference.draw_mask(img, tuple(area))
            score = iou_score(mask, drawn_mask)
            self.assertGreater(score, 0.99)
            
    def test_crop_area(self):
        for _, mask, area in self.dataloader:
            mask = mask.cuda()
            img = torch.cat(3 * [1 - mask])
            cropped_img = self.content_area_inference.crop_area(img, tuple(area), tuple(img.shape[1:]))
            self.assertAlmostEqual(torch.sum(cropped_img).item(), 0, delta=4)

if __name__ == '__main__':
    unittest.main()
