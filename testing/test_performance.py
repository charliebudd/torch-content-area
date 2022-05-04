import unittest
import torch

from utils import TestDataset, TestDataLoader, timed, iou_score, perimeter_distance_score
from torchcontentarea import ContentAreaInference

MISS_THRESHOLD=10
BAD_MISS_THRESHOLD=20

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataloader = TestDataLoader(TestDataset())
        self.content_area_inference = ContentAreaInference()

    def test_infer_mask(self):

        times = []
        errors = []

        for img, mask, area in self.dataloader:
            img, mask = img.cuda(), mask.cuda()

            time, infered_area = timed(lambda x: self.content_area_inference.infer_area(x), img)

            error = perimeter_distance_score(area, infered_area)

            if error > 1000:
                error = 1000

            times.append(time)
            errors.append(error)


        avg_time = sum(times) / len(times)
        avg_error = sum(errors) / len(errors)
        miss_percentage = 100 * sum(map(lambda x: x > MISS_THRESHOLD, errors)) / len(errors)
        bad_miss_percentage = 100 * sum(map(lambda x: x > BAD_MISS_THRESHOLD, errors)) / len(errors)

        gpu_name = torch.cuda.get_device_name()

        print(f'Performance Results...')
        print(f'- Avg Time ({gpu_name}): {avg_time:.3f}ms')
        print(f'- Avg Error (Perimeter Distance): {avg_error:.3f}')
        print(f'- Misses (Error > {MISS_THRESHOLD}): {miss_percentage:.1f}%')
        print(f'- Bad Misses (Error > {BAD_MISS_THRESHOLD}): {bad_miss_percentage:.1f}%')

        self.assertTrue(avg_time < 0.5)
        self.assertTrue(avg_error < 10)
        self.assertTrue(miss_percentage < 10.0)
        self.assertTrue(bad_miss_percentage < 5.0)

if __name__ == '__main__':
    unittest.main()
