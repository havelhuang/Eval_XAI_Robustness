from torchvision import datasets, transforms


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

_FashionMnist_TRAIN_TRANSFORMS = _FashionMnist_TEST_TRANSFORMS = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]

_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]



_CELEBA_TRAIN_TRANSFORMS = _CELEBA_TEST_TRANSFORMS = [
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]


TRAIN_DATASETS = {
    'mnist': datasets.MNIST(
        'Datasets/mnist', train=True, download=False,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        'Datasets/cifar10', train=True, download=False,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'FashionMnist': datasets.FashionMNIST(
        'Datasets/FashionMnist', train=True, download=False,
        transform=transforms.Compose(_FashionMnist_TRAIN_TRANSFORMS)
    ),
    'svhn': datasets.SVHN(
        'Datasets/svhn', split='train', download=False,
        transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS)
    ),
    'celeba': datasets.CelebA(
        root = '../Datasets/', split = "train",download=False,
        transform=transforms.Compose(_CELEBA_TRAIN_TRANSFORMS)
    )
}


TEST_DATASETS = {
    'mnist': datasets.MNIST(
        'Datasets/mnist', train=False, download=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        'Datasets/cifar10', train=False, download=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'FashionMnist': datasets.FashionMNIST(
        'Datasets/FashionMnist', train=False, download=False,
        transform=transforms.Compose(_FashionMnist_TEST_TRANSFORMS)
    ),
    'svhn': datasets.SVHN(
        'Datasets/svhn', split='test', download=False,
        transform=transforms.Compose(_SVHN_TEST_TRANSFORMS)
    ),
    'celeba': datasets.CelebA(
        root = '../Datasets/', split = "test", download=False,
        transform=transforms.Compose(_CELEBA_TEST_TRANSFORMS)
    )
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'FashionMnist': {'size': 32, 'channels': 1, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'celeba': {'size': 64, 'channels': 3, 'classes': 5},
}
