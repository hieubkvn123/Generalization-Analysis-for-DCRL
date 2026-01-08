import os
import time
import tqdm
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloader
from common import apply_model_to_batch, save_json_dict
from model import (
    get_model,
    save_model,
    compute_complexity_ours,
    compute_complexity_ours_opt,
    compute_complexity_bartlett,
    compute_complexity_paracount
)

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
MAX_EPOCHS = 100
BATCH_SIZE = 64
TRAIN_LOSS_THRESHOLD = 0.1 # 0.05 # 1e-2

# Constants for ablation study
MIN_WIDTH = 1
MIN_DEPTH = 2
MAX_WIDTH = MIN_WIDTH + 7
MAX_DEPTH = MIN_DEPTH + 8
DATASET_TO_INDIM = {
  'mnist': 28 * 28,          # 784 for flattened, or use (1, 28, 28) for CNNs
  'fashionmnist': 28 * 28,   # 784 for flattened, or use (1, 28, 28) for CNNs
  'cifar10': 32 * 32 * 3     # 3072 for flattened, or use (3, 32, 32) for CNNs
}
REG_CONSTS  = {2: 0.1, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.005, 7: 0.002, 8: 0.0005, 9: 5e-05, 10: 1e-05}
RESULT_KEYS = {'bartlett': 'Bartlett et al.', 'paracount': 'Graf et al.', 'ours': 'Ours', 'ours_opt': 'Ours (line search)'}
COLOR_KEYS  = {'bartlett': 'tab:orange', 'paracount': 'tab:red', 'ours': 'tab:blue', 'ours_opt': 'tab:cyan'}
SAVE_DIR    = 'checkpoints'

# Create save directory if not available
pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Function to compute L1 norm of all parameters
def compute_l1_norm(model):
    l1_norm = 0.0
    num_params = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param)).item()
        num_params += len(param)
    print(num_params)
    return l1_norm / num_params

# Function to compute L1 regularization loss
def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# Function to compute spectral norm (largest singular value) of a matrix
def compute_spectral_norm(weight_matrix):
    # Using torch.linalg.matrix_norm with ord=2 computes spectral norm
    return torch.linalg.matrix_norm(weight_matrix, ord=2)

# Function to compute spectral norms of all weight matrices
def compute_all_spectral_norms(model):
    spectral_norms = []
    for weight in model.get_weight_matrices():
        spec_norm = compute_spectral_norm(weight)
        spectral_norms.append(spec_norm.item())
    return spectral_norms, sum(spectral_norms)

# Function to compute spectral regularization loss
def spectral_regularization(model, lambda_spectral):
    spectral_loss = 0.0
    for weight in model.get_weight_matrices():
        spectral_loss += compute_spectral_norm(weight)
    return lambda_spectral * spectral_loss

def train(epochs, dataset='mnist', L=2, hidden_dim=128, num_classes=10, batch_size=64, reg_lambda=0.001):
    # Get dataset 
    train_dataloader, test_dataloader = get_dataloader(name=dataset, batch_size=batch_size)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    
    # Load model - output dimension should match number of classes
    model = get_model(in_dim=DATASET_TO_INDIM[dataset], out_dim=num_classes, hidden_dim=hidden_dim, L=L)
    model = model.to(model.device)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0009,
    )
    
    # Loss function for classification
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # To be stored as final result
    final_average_train_loss, final_average_test_loss = 0, 0
    final_train_accuracy, final_test_accuracy = 0, 0

    # Train model
    model.train()
    for epoch in range(epochs):
        print(f'[*] Epoch #[{epoch+1}/{epochs}]:')
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            total_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()

                # Move data to device
                images = images.to(model.device)
                labels = labels.to(model.device)
                
                # Forward pass + CE calculation
                outputs  = model(images)
                ce_loss  = criterion(outputs, labels)
                reg_loss = l1_regularization(model, reg_lambda)
                loss = ce_loss + reg_loss 

                # Back propagation
                loss.backward()
                optimizer.step()

                # Update loss and accuracy for this epoch
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss' : f'{loss.item():.5f}',
                    'batch' : f'#[{i+1}/{num_train_batches}]' 
                })
                pbar.update(1)
            time.sleep(0.1)
            final_average_train_loss = total_loss / (num_train_batches * batch_size)
            final_train_accuracy = 100 * correct / total
            print(f'\nAverage train loss: {final_average_train_loss:.4f}, Accuracy: {final_train_accuracy:.2f}%\n------\n')

        if final_average_train_loss <= TRAIN_LOSS_THRESHOLD:
            print('[INFO] Train loss target reached, early stopping...')
            break

    # Evaluate the model
    model.eval()
    print('------\nEvaluation:')
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dataloader)) as pbar:
            total_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(test_dataloader):
                # Move data to device
                images = images.to(model.device)
                labels = labels.to(model.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update loss and accuracy
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'test_loss' : f'{loss.item():.5f}',
                    'batch' : f'#[{i+1}/{num_test_batches}]'
                })
                pbar.update(1)
            time.sleep(0.1)
            final_average_test_loss = total_loss / (num_test_batches * batch_size)
            final_test_accuracy = 100 * correct / total
            print(f'Average test loss: {final_average_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%')

    # Evaluate complexity measures
    print('------\nComplexity measures computation:')
    cm_ours = np.log(compute_complexity_ours(model, n=len(train_dataloader.dataset)))
    cm_ours_opt = np.log(compute_complexity_ours_opt(model, n=len(train_dataloader.dataset)))
    cm_bartlett = np.log(compute_complexity_bartlett(model, n=len(train_dataloader.dataset)))
    cm_paracount = np.log(compute_complexity_paracount(model, n=len(train_dataloader.dataset)))
    return {
        'ours': cm_ours,
        'ours_opt': cm_ours_opt,
        'bartlett': cm_bartlett,
        'paracount': cm_paracount
    }, model, final_average_train_loss, final_average_test_loss, final_train_accuracy, final_test_accuracy

def ablation_study_varying_depths(args, min_depth, max_depth):
    # Initialize results
    depths = list(range(min_depth, max_depth + 1))
    results_depth = {x: [] for x in list(RESULT_KEYS.keys())} 
    train_losses, test_losses = [], []
    reg_lambdas = {}

    # Conduct training
    for i, L in enumerate(depths):
        print(f'[INFO] Experiment #[{i+1}/{len(depths)}], L = {L}')
        cm, model, train_loss, test_loss, train_acc, test_acc = train(
            epochs=MAX_EPOCHS, 
            batch_size=BATCH_SIZE,
            L=L,
            dataset=args['dataset'],
            hidden_dim=args['hidden_dim'],
            reg_lambda=REG_CONSTS[L]
        )

        # Save results
        for key, item in cm.items():
            results_depth[key].append(item)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Save models
        save_file = os.path.join(SAVE_DIR, f'L{L}.pt')
        save_model(model, save_file)

    return {
        'depths' : depths,
        'complexities' : results_depth,
        'train_loss' : train_losses,
        'test_loss' : test_losses
    }

def ablation_study_varying_widths(args, min_width, max_width):
    # Initialize results
    widths = list(range(min_width, max_width + 1))
    results_width = {x: [] for x in list(RESULT_KEYS.keys())} 
    train_losses, test_losses = [], []

    # Conduct training
    for i, W in enumerate(widths):
        print(f'[INFO] Experiment #[{i+1}/{len(widths)}], W = {W*32}')
        cm, model, train_loss, test_loss, train_acc, test_acc = train(
            epochs=MAX_EPOCHS, 
            batch_size=BATCH_SIZE,
            L=args['L'],
            dataset=args['dataset'],
            hidden_dim=W * 32,
            reg_lambda=REG_CONSTS[args['L']]
        )

        # Save results
        for key, item in cm.items():
            results_width[key].append(item)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Save models
        save_file = os.path.join(SAVE_DIR, f'W{W * 32}.pt')
        save_model(model, save_file)
    
    return {
        'widths' : widths, 
        'complexities' : results_width,
        'train_loss' : train_losses,
        'test_loss' : test_losses
    }

if __name__ == '__main__':
    # Ablation study with depth
    args = {'dataset' : 'mnist', 'hidden_dim' : 256} # Keep hidden dim at 256
    results = ablation_study_varying_depths(args, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
    save_json_dict(results, 'results/ablation_study_depth.json')

    # Ablation study with width
    # args = {'dataset' : 'mnist', 'L' : 3} # Keep depth at 3 layers
    # results = ablation_study_varying_widths(args, min_width=MIN_WIDTH, max_width=MAX_WIDTH)
    # save_json_dict(results, 'results/ablation_study_width.json')
