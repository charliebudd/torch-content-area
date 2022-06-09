import torch
import unittest

from torchcontentarea import ContentAreaInference
from utils import TestDataset, TestDataLoader, timed, mean_border_distance

MISS_THRESHOLD=5
BAD_MISS_THRESHOLD=10

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = TestDataset()
        self.dataloader = TestDataLoader(self.dataset)
        self.content_area_inference = ContentAreaInference()

    def test_circle_accuracy(self):

        times = []
        errors = []

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for img, area in self.dataloader:
            
            img = img.cuda()
            
            time, infered_area = timed(lambda x: self.content_area_inference.infer_area(x), img)

            if area == None:
                if infered_area == None:
                    true_negatives += 1
                else:
                    false_positives += 1
            else:
                if infered_area == None:
                    false_negatives += 1
                else:
                    true_positives += 1

            if area != None and infered_area != None:
                error = mean_border_distance(area, infered_area, img.shape[1:3])
                errors.append(error)

            times.append(time)

        sample_count = len(self.dataset)

        gpu_name = torch.cuda.get_device_name()
        avg_time = sum(times) / sample_count

        classification_accuracy = (true_positives + true_negatives) / sample_count
        fn_rate = false_negatives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 1.0
        fp_rate = false_positives / (true_negatives + false_positives) if true_negatives + false_positives > 0 else 1.0

        avg_error = sum(errors) / len(errors) if len(errors) > 0 else 0.0
        miss_percentage = 100 * sum(map(lambda x: x > MISS_THRESHOLD, errors)) / len(errors) if len(errors) > 0 else 0.0
        bad_miss_percentage = 100 * sum(map(lambda x: x > BAD_MISS_THRESHOLD, errors)) / len(errors) if len(errors) > 0 else 0.0

        total_errors = 100 * (false_negatives + false_positives + sum(map(lambda x: x > BAD_MISS_THRESHOLD, errors))) / sample_count

        print("\n")
        print(f'Performance Results...')
        print(f'- Avg Time ({gpu_name}): {avg_time:.3f}ms')
        print(f'- Avg Error (Mean Perimeter Distance): {avg_error:.3f}px')
        print(f'- Miss Rate (Error > {MISS_THRESHOLD}px): {miss_percentage:.1f}%')
        print(f'- Bad Miss Rate (Error > {BAD_MISS_THRESHOLD}px): {bad_miss_percentage:.1f}%')
        print(f'- Classification Accuracy: {100 * classification_accuracy:.1f}%')
        print(f'- False Negative Rate: {100 * fn_rate:.1f}%')
        print(f'- False Positive Rate: {100 * fp_rate:.1f}%')
        print(f'- Total Error Rate (Bad Misses + Miss-classified): {total_errors:.1f}%')
        print("\n")

        try:
            times = self.content_area_inference.get_times()
            if len(times) > 0:
                print("\nProfiling Data...")
                print("{:<20} {:<20} {:<20} {:<20}".format('section','cuda(μs)','cpu(μs)', 'overall(μs)'))
                for name, time in times:
                    print("{:<20} {:<20} {:<20} {:<20}".format(name, int(time[0]), int(time[1]), int(time[2])))
                print("\n")
        except:
            pass

        self.assertTrue(avg_time < 0.5)
        self.assertTrue(avg_error < 10)
        self.assertTrue(miss_percentage < 10.0)
        self.assertTrue(bad_miss_percentage < 5.0)

if __name__ == '__main__':
    unittest.main()
