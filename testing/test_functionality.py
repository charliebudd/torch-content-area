import torch
import unittest

from utils.data import DummyDataset, TestDataLoader
from utils.scoring import content_area_hausdorff, MISS_THRESHOLD

from torchcontentarea import ContentAreaInference

def iou_score(a, b):
    SMOOTH = 1
    intersection = torch.logical_and(a, b).sum()
    union = torch.logical_or(a, b).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.item()

class TestAPI(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = DummyDataset()
        self.dataloader = TestDataLoader(self.dataset)
        self.content_area_inference = ContentAreaInference()

    def test_infer_area(self):
        for img, mask, area in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            infered_area = self.content_area_inference.infer_area(img)
            distance, _ = content_area_hausdorff(area, infered_area, img.shape[1:]) 
            self.assertLess(distance, MISS_THRESHOLD)

    def test_infer_mask(self):
        for img, mask, _ in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            infered_mask = self.content_area_inference.infer_mask(img)
            score = iou_score(mask, infered_mask)
            self.assertGreaterEqual(round(score, 2), 0.99)

    def test_draw_mask(self):
        for img, mask, area in self.dataloader:
            img, mask = img.cuda(), mask.cuda()
            drawn_mask = self.content_area_inference.draw_mask(img, area)
            score = iou_score(mask, drawn_mask)
            self.assertGreaterEqual(round(score, 2), 0.99)
            
    def test_crop_area(self):
        for _, mask, area in self.dataloader:
            mask = mask.cuda()
            img = torch.cat(3 * [mask])
            cropped_img = self.content_area_inference.crop_area(img, area, tuple(img.shape[1:]))
            self.assertAlmostEqual(torch.sum(cropped_img).item(), 0, delta=4)

if __name__ == '__main__':
    unittest.main()
