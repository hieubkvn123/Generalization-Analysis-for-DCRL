import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# --- Load MNIST and create subset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download full MNIST dataset
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Create subset: 200 samples per class
samples_per_class = 200
num_classes = 10

X_list = []
y_list = []

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
X = X[perm]
y = y[perm]

d = X.shape[1]  # 784 for MNIST
m = num_classes  # 10 classes

print(f"Dataset size: {N} samples, {d} features, {m} classes")

# --- linear model ---
model = nn.Linear(d, m, bias=False)  # no bias for simplicity

# --- training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- sparsity inducing regularization ---
p_reg = 1  # example p value
lambda_reg = 0.0001  # Reduced for MNIST

def lp_regularizer(A, p):
    # A is weight matrix
    return torch.sum(torch.abs(A)**p)

# --- training loop ---
print("Training...")
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    # add reweighted Lp proxy
    reg = lp_regularizer(model.weight, p_reg)
    loss = loss + lambda_reg * reg
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()
            print(f"Epoch {epoch+1}/300, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# --- Compute R (max L2 norm of inputs) ---
with torch.no_grad():
    R = torch.max(torch.norm(X, p=2, dim=1)).item()
    print(f"\nMaximum input L2 norm (R): {R:.4f}")

# --- Compute gamma (MINIMUM classifier margin) ---
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
    gamma = torch.min(margins).item()
    print(f"Minimum classifier margin (gamma): {gamma:.4f}")

# --- compute complexity term for multiple p ---
ps = [0.001, 0.1, 0.25, 0.5, 0.75, 1.0]
C_p = []
with torch.no_grad():
    A = model.weight.data
    md = A.numel()
    for p in ps:
        norm_p = torch.sum(torch.abs(A)**p)**(1/p)
        # New complexity term: gamma^{-p/(p+2)} * [R * ||A||_p * sqrt(md)]^{p/(p+2)}
        exponent = p / (p + 2)
        cp = (gamma ** (-exponent)) * ((R * norm_p * np.sqrt(md)) ** exponent)
        C_p.append(cp.item())

# --- plot ---
plt.figure(figsize=(8, 5))
plt.plot(ps, C_p, marker='o', linewidth=2, markersize=8)
plt.xlabel('p', fontsize=12)
plt.ylabel('Complexity term $\\gamma^{-\\frac{p}{p+2}}[R\\|A\\|_p\\sqrt{md}]^{\\frac{p}{p+2}}$', fontsize=11)
plt.title('Effect of p on theoretical complexity (MNIST subset, min margin)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print final accuracy
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).float().mean().item()
    print(f"\nFinal training accuracy: {accuracy:.4f}")
