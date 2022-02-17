import unittest

from utils import TestDataLoader, timer, iou_score

import torch
from torchcontentarea import get_area_mask

@timer()
def infer_mask(img):
    return get_area_mask(img)

class TestPerformance(unittest.TestCase):

    def test_get_area_mask(self):
      
        dataloader = TestDataLoader()

        times = []
        scores = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, mask = infer_mask(img)
            score = iou_score(mask, seg)

            times.append(time)
            scores.append(score)
            
        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        miss_percentage = 100 * sum(map(lambda x: x < 0.95, scores)) / len(scores)

        self.assertTrue(avg_time < 2.0)
        self.assertTrue(avg_score > 0.95)
        self.assertTrue(miss_percentage < 10.0)

        print(f'')
        print(f'Avg Time: {avg_time:.3f}ms')
        print(f'Avg Score (IoU): {avg_score:.3f}')
        print(f'Misses (IoU < 0.95): {miss_percentage:.3f}%')

if __name__ == '__main__':
    unittest.main()
