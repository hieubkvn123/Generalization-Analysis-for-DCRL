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

    # Complexity from ours (theorem 3)
    complexity = np.log(compute_complexity_ours(model))
    print(f'[INFO] - Complexity measure (ours): {complexity:.4f}\n')

    # Complexity from ours (theorem 3)
    complexity = np.log(compute_complexity_ours_opt(model))
    print(f'[INFO] - Complexity measure (ours - opt): {complexity:.4f}\n')
    
    # Complexity from ours (theorem 2) 
    complexity = np.log(compute_complexity_bartlett(model))
    print(f'[INFO] - Complexity measure (bartlett): {complexity:.4f}\n')
    
    # Complexity from ours (theorem 1) 
    complexity = np.log(compute_complexity_paracount(model))
    print(f'[INFO] - Complexity measure (paracount): {complexity:.4f}\n')
    
