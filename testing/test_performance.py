import unittest
import torch
from torchvision.transforms.functional import center_crop

from utils import TestDataset, TestDataLoader, timed, iou_score, perimeter_distance_score
from torchcontentarea import ContentAreaInference

MISS_THRESHOLD=5
BAD_MISS_THRESHOLD=10

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataloader = TestDataLoader(TestDataset())
        self.content_area_inference = ContentAreaInference()

    def test_circle_accuracy(self):

        times = []
        errors = []

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for img, mask, area in self.dataloader:
            
            img, mask = img.cuda(), mask.cuda()
            img_cropped = center_crop(img, [img.shape[1] // 2, img.shape[2] // 2])

            time, infered_area = timed(lambda x: self.content_area_inference.infer_area(x), img)
            infered_area_cropped = self.content_area_inference.infer_area(img_cropped)

            if infered_area == None:
                false_negatives += 1
            else:
                true_positives += 1
                
            if infered_area_cropped == None:
                true_negatives += 1
            else:
                false_positives += 1

            if infered_area != None:
                error = perimeter_distance_score(area, infered_area)
                errors.append(error)

            times.append(time)

        avg_time = sum(times) / len(times)
        avg_error = sum(errors) / len(errors)
        miss_percentage = 100 * sum(map(lambda x: x > MISS_THRESHOLD, errors)) / len(errors)
        bad_miss_percentage = 100 * sum(map(lambda x: x > BAD_MISS_THRESHOLD, errors)) / len(errors)
        classification_accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        gpu_name = torch.cuda.get_device_name()

        print(f'Performance Results...')
        print(f'- Avg Time ({gpu_name}): {avg_time:.3f}ms')
        print(f'- Avg Error (Perimeter Distance): {avg_error:.3f}px')
        print(f'- Misses (Error > {MISS_THRESHOLD}px): {miss_percentage:.1f}%')
        print(f'- Bad Misses (Error > {BAD_MISS_THRESHOLD}px): {bad_miss_percentage:.1f}%')
        print(f'- Classification Accuracy: {100 * classification_accuracy:.1f}%')

        self.assertTrue(avg_time < 0.5)
        self.assertTrue(avg_error < 10)
        self.assertTrue(miss_percentage < 10.0)
        self.assertTrue(bad_miss_percentage < 5.0)

if __name__ == '__main__':
    unittest.main()
