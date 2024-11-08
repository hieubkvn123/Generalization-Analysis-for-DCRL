import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import os
import pathlib
import numpy as np
from collections import defaultdict
from common import get_default_device

# Data loader definition
class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, k, n):
        super().__init__()

        # Store configurations
        self.k = k
        self.n = n
        self.dataset = dataset
        
        # Mape labels to instance indices
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)
        self.label_to_indices = {label: np.array(indices) for label, indices in self.label_to_indices.items()}

        # Get contrastive tuples
        self.all_labels = list(self.label_to_indices.keys())
        self.all_tuples = self._initialize_data_tuples(n)

    # Sample from the original dataset n tuples of k+2 vectors
    def _initialize_data_tuples(self, n):
        all_tuples = []
        for _ in range(n):
            # Get a random index from original dataset
            index = np.random.randint(0, len(self.dataset))
            _, label = self.dataset[index]

            # Get positive index
            positive_index = np.random.choice(self.label_to_indices[label])
            current_instance = [index, positive_index]

            # Get negative indices
            negative_labels = np.random.choice([l for l in self.all_labels if l != label], self.k, replace=True)
            for neg_label in negative_labels:
                negative_index = np.random.choice(self.label_to_indices[neg_label])
                current_instance.append(negative_index)
            all_tuples.append(current_instance)
        return all_tuples

    def __getitem__(self, index):
        # Get tuple of instance indices
        current_instance = self.all_tuples[index]
        anchor_idx, positive_idx = current_instance[0], current_instance[1]

        # Get positive + anchor instance
        x, _ = self.dataset[anchor_idx]
        x_positive, _ = self.dataset[positive_idx]

        # Flatten
        x = x.view(-1)
        x_positive = x_positive.view(-1)
        
        # Get negative samples 
        negative_samples = []
        for negative_idx in current_instance[2:]:
            x_negative, _ = self.dataset[negative_idx]
            x_negative = x_negative.view(-1)
            negative_samples.append(x_negative)
        return (x, x_positive, negative_samples)

    def __len__(self):
        return len(self.all_tuples)

# Data loader
default_transform = transforms.Compose([transforms.ToTensor()])
def get_dataset(name='cifar100', k=3, n=1000):
    # Get raw dataset
    train_data, test_data = None, None
    if name == 'cifar100':
        train_data = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=default_transform)
    elif name == 'mnist':
        train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=default_transform)

    # Wrap them in custom dataset definition
    train_data = UnsupervisedDataset(train_data, k=k, n=n)
    test_data  = UnsupervisedDataset(test_data, k=k, n=n//3)
    return train_data, test_data

def get_dataloader(name='cifar100', save_path='dataloaders', save_loader=True, batch_size=64, num_batches=1000, sample_ratio=1.0, k=3):
    # Get loader directly if saved
    loader_dir = os.path.join(save_path, name, f'n{num_batches}-k{k}')
    train_path = os.path.join(loader_dir, 'train.pth')
    test_path  = os.path.join(loader_dir, 'test.pth')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('[INFO] Loaders already exist, loading from', loader_dir)
        train_dataloader = torch.load(train_path)
        test_dataloader = torch.load(test_path)
        return train_dataloader, test_dataloader

    # Get dataset
    train_data, test_data = get_dataset(name=name, k=k, n=num_batches*batch_size)

    # Sample fewer data samples
    train_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(train_data) * sample_ratio))
    )
    test_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(test_data) * sample_ratio))
    )

    # Create custom dataloaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=False)

    # Save loader if requested
    pathlib.Path(loader_dir).mkdir(parents=True, exist_ok=True)
    if save_loader:
        torch.save(train_dataloader, os.path.join(loader_dir, 'train.pth'))
        torch.save(test_dataloader, os.path.join(loader_dir, 'test.pth'))
    return train_dataloader, test_dataloader
