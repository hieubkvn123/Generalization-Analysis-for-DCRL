import torch
import numpy as np
from model import (
    get_model,
    compute_complexity_ours,
    compute_complexity_ours_opt,
    compute_complexity_bartlett,
    compute_complexity_paracount
)

if __name__ == '__main__':
    # Initialize model + dataloader
    model = get_model()

    complexity = np.log(compute_complexity_ours(model))
    print(f'[INFO] - Complexity measure (ours - p=0.5): {complexity:.4f}\n')

    complexity = np.log(compute_complexity_ours_opt(model))
    print(f'[INFO] - Complexity measure (ours - line search): {complexity:.4f}\n')
    
    complexity = np.log(compute_complexity_bartlett(model))
    print(f'[INFO] - Complexity measure (bartlett): {complexity:.4f}\n')
    
    complexity = np.log(compute_complexity_paracount(model))
    print(f'[INFO] - Complexity measure (paracount): {complexity:.4f}\n')
    
