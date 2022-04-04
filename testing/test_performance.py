import unittest

from utils import TestDataset, TestDataLoader, timed, iou_score

from torchcontentarea import ContentAreaInference

dataset = TestDataset()
dataloader = TestDataLoader(dataset)
content_area_inference = ContentAreaInference()

class TestPerformance(unittest.TestCase):
                            
    def test_infer_mask(self):

        area_times = []
        mask_times = []
        scores = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            area_time, _ = timed(lambda x: content_area_inference.infer_area(x), img)
            mask_time, mask = timed(lambda x: content_area_inference.infer_mask(x), img)

            score = iou_score(mask, seg)

            area_times.append(area_time)
            mask_times.append(mask_time)
            scores.append(score)
            
        avg_area_time = sum(area_times) / len(area_times)
        avg_mask_time = sum(mask_times) / len(mask_times)
        avg_score = sum(scores) / len(scores)
        miss_percentage = 100 * sum(map(lambda x: x < 0.99, scores)) / len(scores)
        bad_miss_percentage = 100 * sum(map(lambda x: x < 0.95, scores)) / len(scores)

        print(f'Performance Results...')
        print(f'- Avg Time (Infer Area Only): {avg_area_time:.3f}ms')
        print(f'- Avg Time (Infer Area and Draw Mask): {avg_mask_time:.3f}ms')
        print(f'- Avg Score (IoU): {avg_score:.3f}')
        print(f'- Misses (IoU < 0.99): {miss_percentage:.1f}%')
        print(f'- Bad Misses (IoU < 0.95): {bad_miss_percentage:.1f}%')

        self.assertTrue(avg_area_time < 0.3)
        self.assertTrue(avg_mask_time < 0.4)
        self.assertTrue(avg_score > 0.98)
        self.assertTrue(miss_percentage < 10.0)
        self.assertTrue(bad_miss_percentage < 5.0)

if __name__ == '__main__':
    unittest.main()
