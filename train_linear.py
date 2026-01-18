import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from norms import l21_norm, spectral_norm

# --- Load MNIST and create subset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download full MNIST dataset
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create subset: 200 samples per class
samples_per_class = 4000
num_classes = 10

# Prepare training data
X_list, y_list = [], []
for class_idx in range(num_classes):
    # Find all indices for this class
    class_indices = [i for i, (_, label) in enumerate(mnist_train) if label == class_idx]
    # Select first samples_per_class instances
    selected_indices = class_indices[:samples_per_class]

    for idx in selected_indices:
        img, label = mnist_train[idx]
        X_list.append(img.flatten())  # Flatten 28x28 to 784
        y_list.append(label)
X = torch.stack(X_list)
y = torch.tensor(y_list, dtype=torch.long)

# Shuffle the data
N = len(y)
perm = torch.randperm(N)
X, y = X[perm], y[perm]
d, m = X.shape[1], num_classes
print(f"Training dataset size: {N} samples, {d} features, {m} classes")

# Prepare test dataset
X_test_list, y_test_list = [], []
for img, label in mnist_test:
    X_test_list.append(img.flatten())
    y_test_list.append(label)
X_test = torch.stack(X_test_list)
y_test = torch.tensor(y_test_list, dtype=torch.long)
print(f"Test dataset size: {len(y_test)} samples")

# --- sparsity inducing regularization ---
def lp_regularizer(A, p):
    # A is weight matrix
    return torch.sum(torch.abs(A)**p)

# --- Prune sparse matrices ---
def prune_matrix(matrix, threshold=1e-3):
    pruned_matrix = matrix.copy()
    pruned_matrix[np.abs(pruned_matrix) < threshold] = 0
    return pruned_matrix

# --- Compute gamma (margin threshold for target accuracy) ---
def compute_margin_threshold(model, X, y, target_accuracy=0.85):
    N = len(y)
    with torch.no_grad():
        outputs = model(X)
        # Get predicted class scores
        pred_scores = outputs[torch.arange(N), y]  # scores for true class
        # Get max score among other classes
        outputs_copy = outputs.clone()
        outputs_copy[torch.arange(N), y] = -float('inf')
        max_other_scores = torch.max(outputs_copy, dim=1)[0]
        # Margin = score(true class) - max(score(other classes))
        margins = pred_scores - max_other_scores

        # Check if predictions are correct
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y)

        # Get margins only for correctly classified samples
        correct_margins = margins[correct]

        # Sort margins in ascending order
        sorted_margins = torch.sort(correct_margins)[0]

        # Find the margin threshold that gives us target_accuracy
        num_correct = correct.sum().item()
        num_needed = int(np.ceil(target_accuracy * N))

        if num_correct >= num_needed:
            # Index of the margin threshold (sorted in ascending order)
            threshold_idx = max(0, num_correct - num_needed)
            gamma = sorted_margins[threshold_idx].item()
            print(f"Margin threshold (gamma) for {target_accuracy*100}% accuracy: {gamma:.4f}")
            print(f"Number of samples with margin >= gamma: {(margins >= gamma).sum().item()}/{N}")
        else:
            # Not enough correct predictions to meet target accuracy
            gamma = torch.min(correct_margins).item()
            print(f"Warning: Only {num_correct}/{N} correct predictions, cannot achieve {target_accuracy*100}% accuracy")
            print(f"Using minimum margin among correct predictions: {gamma:.4f}")

    return gamma

# --- Configs ---
p_reg = 1
lambda_reg = 0.0005
epochs = 1000
ps = [0, 0.15, 0.30, 0.45, 0.65, 0.85, 1.0]

if __name__ == '__main__':
    # --- linear model ---
    model = nn.Linear(d, m, bias=False)

    # --- training setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, amsgrad=True)

    # --- training loop ---
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        # add reweighted Lp proxy
        reg = lp_regularizer(model.weight, p_reg)
        loss = loss + lambda_reg * reg
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    # --- Evaluate on training and test sets ---
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)

    with torch.no_grad():
        # Training accuracy
        outputs_train = model(X)
        _, predicted_train = torch.max(outputs_train, 1)
        train_accuracy = (predicted_train == y).float().mean().item()

        # Test accuracy
        outputs_test = model(X_test)
        _, predicted_test = torch.max(outputs_test, 1)
        test_accuracy = (predicted_test == y_test).float().mean().item()

        # Generalization gap
        generalization_gap = train_accuracy - test_accuracy

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Generalization gap: {generalization_gap:.4f}")

    # --- Compute R (max L2 norm of inputs) ---
    with torch.no_grad():
        R = torch.max(torch.norm(X, p=2, dim=1)).item()
        print(f"\nMaximum input L2 norm (R): {R:.4f}")
    target_accuracy = 0.90
    gamma = compute_margin_threshold(model, X, y, target_accuracy)

    # --- Compute Bartlett et al. complexity ---
    bartlett = l21_norm(A) / (gamma * np.sqrt(N))

    # --- compute complexity term for multiple p ---
    C_p = []
    with torch.no_grad():
        A = model.weight.data.numpy()
        A = prune_matrix(A)
        md = np.prod(A.shape)
        for p in ps:
            if p == 0:
                # L0 norm: count of non-zero entries
                cp = np.count_nonzero(A) / np.sqrt(N)
            else:
                norm_p = np.sum(np.abs(A)**p)**(1/p)
                exponent = p / (p + 2)
                cp = (gamma ** (-exponent)) * ((R * norm_p * np.sqrt(md)) ** exponent)
                cp = cp / np.sqrt(N)
                cp = cp.item()
            C_p.append(cp)

    # --- plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(ps, C_p, marker='o', linewidth=2, markersize=8, label='Complexity term')

    # -- annotate ---
    for p, cp in zip(ps, C_p):
      plt.annotate(
          f"{cp:.2f}",          # or f"{cp:.3f}" if you prefer
          (p, cp),
          textcoords="offset points",
          xytext=(0, 6),        # move text slightly above the marker
          ha="center",
          fontsize=9
      )

    # Add horizontal line for generalization gap
    plt.axhline(y=generalization_gap, color='tab:red', linestyle='--', linewidth=2,
                label=f'Generalization gap ({generalization_gap:.4f})')
    plt.axhline(y=bartlett, color='tab:orange', linestyle='-.', linewidth=2,
                label=f'Bartlett et al. ({bartlett:.4f})')
    plt.xlabel('p', fontsize=12)
    plt.ylabel('Complexity term', fontsize=11)
    plt.title('Effect of p on theoretical complexity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/linear_result.pdf', dpi=300, format='pdf')
