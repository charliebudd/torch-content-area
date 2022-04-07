import unittest
import torch

from utils import TestDataset, TestDataLoader, timed, iou_score
from torchcontentarea import ContentAreaInference

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataloader = TestDataLoader(TestDataset())
        self.content_area_inference = ContentAreaInference()

    def test_infer_mask(self):

        times = []
        scores = []

        for img, seg in self.dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, mask = timed(lambda x: self.content_area_inference.infer_mask(x), img)

            score = iou_score(mask, seg)

            times.append(time)
            scores.append(score)
            
        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        miss_percentage = 100 * sum(map(lambda x: x < 0.99, scores)) / len(scores)
        bad_miss_percentage = 100 * sum(map(lambda x: x < 0.95, scores)) / len(scores)

        gpu_name = torch.cuda.get_device_name()

        print(f'Performance Results...')
        print(f'- Avg Time ({gpu_name}): {avg_time:.3f}ms')
        print(f'- Avg Score (IoU): {avg_score:.3f}')
        print(f'- Misses (IoU < 0.99): {miss_percentage:.1f}%')
        print(f'- Bad Misses (IoU < 0.95): {bad_miss_percentage:.1f}%')

        self.assertTrue(avg_time < 0.3)
        self.assertTrue(avg_score > 0.98)
        self.assertTrue(miss_percentage < 10.0)
        self.assertTrue(bad_miss_percentage < 5.0)

if __name__ == '__main__':
    unittest.main()
