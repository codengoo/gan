from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __int__(
            self,
            root: Union[str, Path].root,
    ):
        super().__init__(root, download=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])


class CIFAR100(datasets.CIFAR100):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __int__(
            self,
            root: Union[str, Path].root
    ):
        super().__init__(root, download=True)

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])
