import unittest

from utils import TestDataLoader, timer, iou_score

from torchcontentarea import ContentAreaInference


content_area_inference = ContentAreaInference()

@timer()
def infer_area_timed(img):
    content_area_inference.infer_area(img)
    return None

@timer()
def draw_mask_timed(img):
    content_area_inference.draw_mask(img, ("Circle", (240, 240, 240)))
    return None

@timer()
def infer_area_and_draw_mask_timed(img):
    area = content_area_inference.infer_area(img)
    mask = content_area_inference.draw_mask(img, area)
    return mask
    
@timer()
def infer_mask_timed(img):
    return content_area_inference.infer_mask(img)
    

class TestPerformance(unittest.TestCase):
                
    def test_infer_area(self):
      
        dataloader = TestDataLoader()

        times = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, _ = infer_area_timed(img)

            times.append(time)
            
        avg_time = sum(times) / len(times)

        print(f'')
        print(f'infer_area...')
        print(f'Avg Time: {avg_time:.3f}ms')

    def test_draw_mask(self):
      
        dataloader = TestDataLoader()

        times = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, _ = draw_mask_timed(img)

            times.append(time)
            
        avg_time = sum(times) / len(times)

        print(f'')
        print(f'draw_mask...')
        print(f'Avg Time: {avg_time:.3f}ms')

    def test_infer_area_and_draw_mask(self):
      
        dataloader = TestDataLoader()

        times = []
        scores = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, mask = infer_area_and_draw_mask_timed(img)
            score = iou_score(mask, seg)

            times.append(time)
            scores.append(score)
            
        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        miss_percentage = 100 * sum(map(lambda x: x < 0.95, scores)) / len(scores)

        print(f'')
        print(f'infer_area and draw_mask...')
        print(f'Avg Time: {avg_time:.3f}ms')
        print(f'Avg Score (IoU): {avg_score:.3f}')
        print(f'Misses (IoU < 0.95): {miss_percentage:.1f}%')

        self.assertTrue(avg_time < 2.0)
        self.assertTrue(avg_score > 0.95)
        self.assertTrue(miss_percentage < 10.0)

    def test_infer_mask(self):
      
        dataloader = TestDataLoader()

        times = []
        scores = []

        for img, seg in dataloader:
            img, seg = img.cuda(), seg.cuda()

            time, mask = infer_mask_timed(img)
            score = iou_score(mask, seg)

            times.append(time)
            scores.append(score)
            
        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        miss_percentage = 100 * sum(map(lambda x: x < 0.95, scores)) / len(scores)

        print(f'')
        print(f'infer_mask...')
        print(f'Avg Time: {avg_time:.3f}ms')
        print(f'Avg Score (IoU): {avg_score:.3f}')
        print(f'Misses (IoU < 0.95): {miss_percentage:.1f}%')

        self.assertTrue(avg_time < 2.0)
        self.assertTrue(avg_score > 0.95)
        self.assertTrue(miss_percentage < 10.0)

if __name__ == '__main__':
    unittest.main()
