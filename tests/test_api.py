import torch
import unittest

from .utils.data import DummyDataset
from .utils.scoring import content_area_hausdorff, MISS_THRESHOLD

from torchcontentarea import estimate_area_handcrafted, estimate_area_learned, get_points_handcrafted, get_points_learned, fit_area


ESTIMATION_MEHTODS = [
    ("handcrafted cpu", estimate_area_handcrafted, "cpu"),
    ("learned cpu", estimate_area_learned, "cpu"),
    ("handcrafted cuda", estimate_area_handcrafted, "cuda"),
    ("learned cuda", estimate_area_learned, "cuda"),
    ("handcrafted cpu two staged", lambda x: fit_area(get_points_handcrafted(x), x.shape[-2:]), "cpu"),
    ("learned cpu two staged", lambda x: fit_area(get_points_learned(x), x.shape[-2:]), "cpu"),
    ("handcrafted cuda two staged", lambda x: fit_area(get_points_handcrafted(x), x.shape[-2:]), "cuda"),
    ("learned cuda two staged", lambda x: fit_area(get_points_learned(x), x.shape[-2:]), "cuda"),
]

class TestAPI(unittest.TestCase):
                            
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = DummyDataset()

    def test_unbatched(self):
        image, _, true_area = self.dataset[0]
        for name, method, device in ESTIMATION_MEHTODS:
            with self.subTest(name):
                image = image.to(device)
                result = method(image).tolist()
                estimated_area = result[0:3]
                error, _ = content_area_hausdorff(true_area, estimated_area, image.shape[-2:])
                self.assertLess(error, MISS_THRESHOLD)
               
    def test_batched(self):
        image_a, _, true_area_a = self.dataset[0]
        image_b, _, true_area_b = self.dataset[1]
        images = torch.stack([image_a, image_b])
        true_areas = [true_area_a, true_area_b]
        for name, method, device in ESTIMATION_MEHTODS:
            with self.subTest(name):
                images = images.to(device)
                results = method(images).tolist()
                for true_area, result in zip(true_areas, results):
                    estimated_area = result[0:3]
                    error, _ = content_area_hausdorff(true_area, estimated_area, images.shape[-2:])
                    self.assertLess(error, MISS_THRESHOLD)
        
    def test_rgb(self):
        image, _, true_area = self.dataset[0]
        for name, method, device in ESTIMATION_MEHTODS:
            with self.subTest(name):
                image = image.to(device)
                result = method(image).tolist()
                estimated_area = result[0:3]
                error, _ = content_area_hausdorff(true_area, estimated_area, image.shape[-2:])
                self.assertLess(error, MISS_THRESHOLD)
    
    def test_grayscale(self):
        image, _, true_area = self.dataset[0]
        image = (0.2126 * image[0:1] + 0.7152 * image[1:2]+ 0.0722 * image[2:3]).to(torch.uint8)
        for name, method, device in ESTIMATION_MEHTODS:
            with self.subTest(name):
                image = image.to(device)
                result = method(image).tolist()
                estimated_area = result[0:3]
                error, _ = content_area_hausdorff(true_area, estimated_area, image.shape[-2:])
                self.assertLess(error, MISS_THRESHOLD)
    
    
    # def test_large(self):
    #     pass
    
    # def test_small(self):
    #     pass
    
    # def test_byte(self):
    #     pass
    
    # def test_int(self):
    #     pass

    # def test_float(self):
    #     pass
    
    # def test_double(self):
    #     pass
    

if __name__ == '__main__':
    unittest.main()
