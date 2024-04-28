from torchvision import transforms, datasets
from pathlib import Path
from typing import List, Union, Literal

TargetType = Literal["attr", "identity", "bbox", "landmarks"]


class CelebA(datasets.CelebA):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __init__(
            self,
            root: Union[str, Path],
            image_size: int = 64,
            target_type: Union[List[TargetType], TargetType] = "attr"):
        super().__init__(root, target_type=target_type, download=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
