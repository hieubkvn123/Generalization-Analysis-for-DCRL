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
    load_model,
    compute_complexity_ours,
    compute_complexity_bartlett,
    compute_complexity_neyshabur,
    compute_complexity_ledent,
    compute_complexity_rank_sparse,
    compute_complexity_paracount,
    compute_complexity_paracount_nonzero
)

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
MAX_EPOCHS = 1000
BATCH_SIZE = 64
TARGET_ACCURACY = 0.92
TRAIN_LOSS_THRESHOLD = 0.05 # 1e-2

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
RESULT_KEYS = {'bartlett': 'Bartlett et al.', 'paracount': 'Graf et al.', 
               'neyshabur': 'Neyshabur et al.',
               'ledent_p0': 'Ledent et al. ($p=0.0$)', 'ledent_p5': 'Ledent et al. ($p=0.5$)',
               'ours_p0': 'Ours ($p=0.0$)', 'ours_p5': 'Ours ($p=0.5$)'}
COLOR_KEYS  = {'bartlett': 'tab:orange', 'paracount': 'tab:red', 
               'neyshabur': 'tab:purple',
               'ledent_p0': 'tab:pink', 'ledent_p5': 'tab:brown',
               'ours_p0': 'tab:blue', 'ours_p5': 'tab:green'}
SAVE_DIR    = 'checkpoints_nsr'

def compute_margin_threshold(all_margins, all_correct, total, target_accuracy=0.85):
    # Concatenate all batches
    all_margins = torch.cat(all_margins)
    all_correct = torch.cat(all_correct)
    
    # Get margins only for correctly classified samples
    correct_margins = all_margins[all_correct]
    
    # Sort margins in ascending order
    sorted_margins = torch.sort(correct_margins)[0]
    
    # Find the margin threshold that gives us target_accuracy
    num_correct = all_correct.sum().item()
    num_needed = int(np.ceil(target_accuracy * total))
    
    if num_correct >= num_needed:
        # Index of the margin threshold (sorted in ascending order)
        threshold_idx = max(0, num_correct - num_needed)
        gamma = sorted_margins[threshold_idx].item()
        print(f"Margin threshold (gamma) for {target_accuracy*100}% accuracy: {gamma:.4f}")
        print(f"Number of samples with margin >= gamma: {(all_margins >= gamma).sum().item()}/{total}")
    else:
        # Not enough correct predictions to meet target accuracy
        gamma = torch.min(correct_margins).item()
        print(f"Warning: Only {num_correct}/{total} correct predictions, cannot achieve {target_accuracy*100}% accuracy")
        print(f"Using minimum margin among correct predictions: {gamma:.4f}")
    return gamma

def evaluate(model_file, dataset='mnist'):
    # Get dataset 
    train_dataloader, test_dataloader = get_dataloader(name=dataset, batch_size=BATCH_SIZE)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    
    # Load model 
    model = load_model(model_file)
    criterion = torch.nn.MultiMarginLoss(p=1, margin=1.0, reduction='sum')

    # To be stored as final result
    final_average_train_loss, final_average_test_loss = 0, 0
    final_train_accuracy, final_test_accuracy = 0, 0

    # Lists to accumulate margins and correctness for gamma computation
    all_margins, all_correct = [], []

    # Evaluate model 
    print('------\nLoss computation on training data:')
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            total_loss, correct, total = 0.0, 0, 0
            for i, (images, labels) in enumerate(train_dataloader):
                # Move data to device
                images = images.to(model.device)
                labels = labels.to(model.device)
                
                # Forward pass + CE calculation
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Compute margins for gamma calculation
                batch_size = labels.size(0)
                pred_scores = outputs[torch.arange(batch_size), labels]  # scores for true class
                outputs_copy = outputs.clone()
                outputs_copy[torch.arange(batch_size), labels] = -float('inf')
                max_other_scores = torch.max(outputs_copy, dim=1)[0]
                margins = pred_scores - max_other_scores  # margin for each sample
                
                # Check predictions
                _, predicted = torch.max(outputs.data, 1)
                correct_batch = (predicted == labels)
                
                # Store margins and correctness
                all_margins.append(margins.cpu())
                all_correct.append(correct_batch.cpu())

                # Update loss and accuracy for this epoch
                total_loss += loss.item()
                total += labels.size(0)
                correct += correct_batch.sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss' : f'{loss.item():.5f}',
                    'batch' : f'#[{i+1}/{num_train_batches}]' 
                })
                pbar.update(1)
            time.sleep(0.1)
            final_average_train_loss = total_loss / (num_train_batches * BATCH_SIZE)
            final_train_accuracy = 100 * correct / total
            print(f'\nAverage train loss: {final_average_train_loss:.4f}, Accuracy: {final_train_accuracy:.2f}%\n------\n')
    gamma = compute_margin_threshold(all_margins, all_correct, total, target_accuracy=TARGET_ACCURACY)
    print(f'Gamma = {gamma}')

    # Evaluate the model
    model.eval()
    print('------\nLoss computation on testing data:')
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dataloader)) as pbar:
            total_loss, correct, total = 0.0, 0, 0
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
            final_average_test_loss = total_loss / (num_test_batches * BATCH_SIZE)
            final_test_accuracy = 100 * correct / total
            print(f'Average test loss: {final_average_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%')

    # Evaluate complexity measures
    print('------\nComplexity measures computation:')
    n = len(train_dataloader.dataset)
    cm_ours_p0   = np.log(compute_complexity_paracount_nonzero(model, n=n))
    cm_ours_p5   = np.log(compute_complexity_ours(model, p=0.6, gamma=gamma, n=n))
    cm_paracount = np.log(compute_complexity_paracount(model, n=n))
    cm_ledent_p0 = np.log(compute_complexity_rank_sparse(model, n=n))
    cm_ledent_p5 = np.log(compute_complexity_ledent(model, p=0.5, gamma=gamma, n=n))
    cm_bartlett  = np.log(compute_complexity_bartlett(model, gamma=gamma, n=n))
    cm_neyshabur = np.log(compute_complexity_neyshabur(model, gamma=gamma, n=n))
    return {
        'ours_p0': cm_ours_p0,
        'ours_p5': cm_ours_p5,
        'ledent_p0': cm_ledent_p0,
        'ledent_p5': cm_ledent_p5,
        'bartlett': cm_bartlett,
        'neyshabur': cm_neyshabur,
        'paracount': cm_paracount
    }, model, final_average_train_loss, final_average_test_loss, final_train_accuracy, final_test_accuracy

def ablation_study_varying_depths(min_depth, max_depth):
    # Initialize results
    depths = list(range(min_depth, max_depth + 1))
    results_depth = {x: [] for x in list(RESULT_KEYS.keys())} 
    train_losses, test_losses = [], []

    # Conduct training
    for i, L in enumerate(depths):
        print(f'\n======\n[INFO] Experiment #[{i+1}/{len(depths)}], L = {L}')
        cm, model, train_loss, test_loss, train_acc, test_acc = evaluate(
            model_file=f'{SAVE_DIR}/L{L}.pt',
            dataset='mnist'
        )

        # Save results
        for key, item in cm.items():
            results_depth[key].append(item)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return {
        'depths' : depths,
        'complexities' : results_depth,
        'train_loss' : train_losses,
        'test_loss' : test_losses
    }

def ablation_study_varying_widths(min_width, max_width):
    # Initialize results
    widths = list(range(min_width, max_width + 1))
    results_width = {x: [] for x in list(RESULT_KEYS.keys())} 
    train_losses, test_losses = [], []

    # Conduct training
    for i, W in enumerate(widths):
        print(f'\n======\n[INFO] Experiment #[{i+1}/{len(widths)}], W = {W*32}')
        cm, model, train_loss, test_loss, train_acc, test_acc = evaluate(
            model_file=f'{SAVE_DIR}/W{W * 32}.pt',
            dataset='mnist'
        )

        # Save results
        for key, item in cm.items():
            results_width[key].append(item)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    return {
        'widths' : widths, 
        'complexities' : results_width,
        'train_loss' : train_losses,
        'test_loss' : test_losses
    }

if __name__ == '__main__':
    # Ablation study with depth
    results = ablation_study_varying_depths(min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
    save_json_dict(results, 'results/ablation_study_depth.json')

    # Ablation study with width
    results = ablation_study_varying_widths(min_width=MIN_WIDTH, max_width=MAX_WIDTH)
    save_json_dict(results, 'results/ablation_study_width.json')

