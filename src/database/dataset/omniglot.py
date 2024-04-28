from torchvision import datasets, transforms


class Omniglot(datasets.Omniglot):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __int__(
            self,
            root: Union[str, Path].root,
            normalization: bool = False
    ):
        super().__init__(root, download=True)

        self.transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            self.normalize
        ])
