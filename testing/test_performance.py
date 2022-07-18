import torch
import unittest

from utils.data import TestDataset, TestDataLoader
from utils.scoring import content_area_hausdorff, MISS_THRESHOLD, BAD_MISS_THRESHOLD
from utils.profiling import Timer

from torchcontentarea import ContentAreaInference, FeatureExtraction

TEST_LOG = ""

MODES = [FeatureExtraction.HANDCRAFTED, FeatureExtraction.LEARNED]
MODE_NAMES = ["handcrafted", "learned"]

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = TestDataset()
        self.dataloader = TestDataLoader(self.dataset)
        self.content_area_inference = ContentAreaInference()

    def test_performance(self):

        times = [[], []]
        errors = [[], []]

        for img, area in self.dataloader:

            img = img.cuda()
            
            for i, mode in enumerate(MODES):

                with Timer() as timer:
                    infered_area = self.content_area_inference.infer_area(img, mode)
                time = timer.time

                error, _ = content_area_hausdorff(area, infered_area, img.shape[1:3])
                errors[i].append(error)
                times[i].append(time)

        for name, times, errors in zip(MODE_NAMES, times, errors):

            gpu_name = torch.cuda.get_device_name()
            run_in_count = int(len(times) // 100)
            times = times[run_in_count:]
            avg_time = sum(times) / len(times)

            sample_count = len(self.dataset)
            average_error = sum(errors) / sample_count
            miss_percentage = 100 * sum(map(lambda x: x > MISS_THRESHOLD, errors)) / sample_count
            bad_miss_percentage = 100 * sum(map(lambda x: x > BAD_MISS_THRESHOLD, errors)) / sample_count

            global TEST_LOG
            TEST_LOG += "\n".join([
                f"\n",
                f"Performance Results ({name})...",
                f"- Avg Time ({gpu_name}): {avg_time:.3f}ms",
                f"- Avg Error (Hausdorff Distance): {average_error:.3f}",
                f"- Miss Rate (Error > {MISS_THRESHOLD}): {miss_percentage:.1f}%",
                f"- Bad Miss Rate (Error > {BAD_MISS_THRESHOLD}): {bad_miss_percentage:.1f}%"
            ])

            self.assertTrue(avg_time < 10)
            self.assertTrue(average_error < 10)
            self.assertTrue(miss_percentage < 10.0)
            self.assertTrue(bad_miss_percentage < 5.0)

    @classmethod
    def tearDownClass(cls):
        if TEST_LOG != "":
            print(TEST_LOG)

if __name__ == '__main__':
    unittest.main()
