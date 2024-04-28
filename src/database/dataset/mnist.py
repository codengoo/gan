import torch
from torchvision import datasets, transforms
from typing import Union
from pathlib import Path


class MNIST(datasets.MNIST):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __init__(
            self,
            root: Union[str, Path].root,
            normalization: bool = False
    ):
        super().__init__(root, download=True, train=True)

        if normalization:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])

        self.transforms = transform


class BigMNIST(datasets.MNIST):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __init__(
            self,
            root: Union[str, Path].root,
            normalization: bool = False
    ):
        super().__init__(root, download=True, train=True)

        if normalization:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
                transforms.Lambda(lambda img: torch.cat([img, img, img], dim=0)),
                transforms.Pad(padding=(2, 2, 2, 2), fill=0, padding_mode="constant")
            ])

        self.transforms = transform


class FashionMNIST(datasets.FashionMNIST):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __int__(
            self,
            root: Union[str, Path].root,
            normalization: bool = False
    ):
        super().__init__(root, download=True, train=True)

        if normalization:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        self.transforms = transform


class DoubleMNIST(datasets.FashionMNIST):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __init__(
            self,
            root: Union[str, Path].root,
            normalization: bool = False
    ):
        super().__init__(root, download=True, train=True)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
            transforms.Lambda(lambda img: torch.cat([img, img, img], dim=0)),
            transforms.Pad(padding=(2, 2, 2, 2), fill=0, padding_mode="constant")
        ])
