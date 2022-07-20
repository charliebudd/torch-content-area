import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from eca import ECADataset, DataSource, AnnotationType

def meshgrid(tensors):
    major, minor = map(int, torch.__version__.split(".")[:2])
    if major >= 1 and minor > 9:
        return torch.meshgrid(tensors, indexing="ij")
    else:
        return torch.meshgrid(tensors)


########################
# Datasets...

class TestDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=False) -> None:
        super().__init__(dataset=dataset, batch_size=None, num_workers=10, pin_memory=True, shuffle=shuffle)


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = ECADataset("eca-data", DataSource.CHOLEC, AnnotationType.AREA, include_cropped=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, area = self.dataset[index]
        img = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return img, area


class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.width = 854
        self.height = 480

        self.areas = [
            (400, 250, 360),
            (340, 200, 370),
            (450, 230, 250),
            None,
        ]

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, index):
        area = self.areas[index]

        if area != None:
            area_x, area_y, area_r = self.areas[index]
            coords = torch.stack(meshgrid([torch.arange(0, self.height), torch.arange(0, self.width)]))
            center = torch.Tensor([area_y, area_x]).reshape((2, 1, 1))
            mask = torch.where(torch.linalg.norm(abs(coords - center), dim=0) < area_r, 0, 1).unsqueeze(0)
        else:
            mask = torch.zeros(1, self.height, self.width)

        img = 255 * (1 - mask).expand((3, self.height, self.width))
        img = img.to(dtype=torch.uint8).contiguous()
        mask = mask.to(dtype=torch.uint8).contiguous()

        return img, mask, area
