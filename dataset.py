import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Input dimensions
DATASET_TO_INDIM = {
    'mnist': 28 * 28,          # 784 for flattened, or use (1, 28, 28) for CNNs
    'fashionmnist': 28 * 28,   # 784 for flattened, or use (1, 28, 28) for CNNs
    'cifar10': 32 * 32 * 3     # 3072 for flattened, or use (3, 32, 32) for CNNs
}

def get_dataloader(name='mnist', batch_size=64, num_batches=None):
    """
    Get train and test dataloaders for specified dataset.

    Args:
        name: Dataset name ('mnist', 'cifar10', 'fashionmnist')
        batch_size: Batch size for dataloaders
        num_batches: Optional limit on number of batches (None for full dataset)

    Returns:
        train_dataloader, test_dataloader
    """
    name = name.lower()

    # Define transforms based on dataset
    if name == 'mnist' or name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel
        ])
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Three channels
        ])
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported datasets: 'mnist', 'cifar10', 'fashionmnist'")

    # Load dataset
    if name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    elif name == 'fashionmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    elif name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    # Optionally limit dataset size based on num_batches
    if num_batches is not None:
        train_size = min(num_batches * batch_size, len(train_dataset))
        test_size = min(num_batches * batch_size, len(test_dataset))

        train_indices = list(range(train_size))
        test_indices = list(range(test_size))

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_dataloader, test_dataloader


