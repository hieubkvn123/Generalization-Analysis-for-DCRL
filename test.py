import torch
import numpy as np
from model import (
    get_model,
    compute_complexity_YW,
    compute_complexity_THM1,
    compute_complexity_THM2,
    compute_complexity_THM3
)

# Constants
dataloader_path = r'dataloaders/mnist/n100-k3/train.pth'

if __name__ == '__main__':
    # Initialize model + dataloader
    model = get_model()
    dataloader = torch.load(dataloader_path)

    # Complexity from ours (theorem 3)
    complexity = np.log(compute_complexity_THM3(dataloader, model))
    print(f'[INFO] Complexity measure from theorem 3: {complexity:.4f}')
    
    # Complexity from ours (theorem 2) 
    complexity = np.log(compute_complexity_THM2(dataloader, model))
    print(f'[INFO] Complexity measure from theorem 2: {complexity:.4f}')
    
    # Complexity from ours (theorem 1) 
    complexity = np.log(compute_complexity_THM1(dataloader, model))
    print(f'[INFO] Complexity measure from theorem 1: {complexity:.4f}')
    
    # Complexity from Yunwen et. al.
    complexity = np.log(compute_complexity_YW(dataloader, model))
    print(f'[INFO] Complexity measure from Yunwen: {complexity:.4f}')
