import torch
import unittest
from time import sleep

from .utils.data import TestDataset, TestDataLoader
from .utils.scoring import content_area_hausdorff, MISS_THRESHOLD, BAD_MISS_THRESHOLD
from .utils.profiling import Timer

from torchcontentarea import infer_area_handcrafted, infer_area_learned

TEST_LOG = ""

TEST_CASES = [
    ("handcrafted cpu", infer_area_handcrafted, "cpu"),
    ("learned cpu", infer_area_learned, "cpu"),
    ("handcrafted cuda", infer_area_handcrafted, "cuda"),
    ("learned cuda", infer_area_learned, "cuda"),
]

class TestPerformance(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = TestDataset()
        self.dataloader = TestDataLoader(self.dataset)

    def test_performance(self):

        times = [[] for _ in range(len(TEST_CASES))]
        errors = [[] for _ in range(len(TEST_CASES))]

        for img, area in self.dataloader:

            img = img.unsqueeze(0)

            for i, (name, method, device) in enumerate(TEST_CASES):

                img = img.to(device=device)

                with Timer() as timer:
                    infered_area = method(img)
                time = timer.time

                infered_area = infered_area[0].cpu().numpy()

                infered_area, confidence = tuple(infered_area[0:3]), infered_area[-1]
                infered_area = tuple(map(int, infered_area))
                if confidence < 0.06:
                    infered_area = None

                error, _ = content_area_hausdorff(area, infered_area, img.shape[2:4])

                errors[i].append(error)
                times[i].append(time)


        for (name, _, _), times, errors in zip(TEST_CASES, times, errors):

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
            sleep(3)
            print(TEST_LOG)

if __name__ == '__main__':
    unittest.main()
