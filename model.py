import torch
import torch.nn as nn

import tqdm
import numpy as np

from norms import frobenius_norm, l21_norm, spectral_norm 
from common import get_default_device, apply_model_to_batch

# Network definition
class Net(nn.Module):
    def __init__(self, in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
        super().__init__()
        
        # Store configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.L = L

        # Store device
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device
        self.device_type = self.device.type

        # Convert model to own device
        self.to(self.device)
        
        # Create layers
        self.fc_hidden_layers = []
        for _ in range(1, self.L):
            self.fc_hidden_layers.append(
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            )
            self.fc_hidden_layers.append(
                nn.ReLU()    
            )
        self.v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(), 
            *self.fc_hidden_layers
        )
        self.U = nn.Linear(hidden_dim, out_dim)

        # Store reference matrices
        self.references = []
        for l in range(1, self.L + 1):
            self.references.append(self._get_v_layer_weights(layer=l))
        
    def _tensor_to_numpy(self, x):
        if self.device_type == 'cuda':
            return x.cpu().detach().numpy()            
        else:
            return x.detach().numpy()
        
    def _get_v_layer_linear(self, layer=1):
        return list(self.v.modules())[0][layer*2-2]
    
    def _get_v_layer_activation(self, layer=1):
        return list(self.v.modules())[0][layer*2-1]
    
    def _get_v_layer_weights(self, layer=1):
        v_layer = self._get_v_layer_linear(layer=layer)
        return self._tensor_to_numpy(v_layer.weight)
    
    def _get_output_from_layer(self, x, last_layer=1, preactivation=False):
        for l in range(1, last_layer + 1):
            # Get the preactivation of current layer
            x = self._get_v_layer_linear(layer=l)(x)

            # Get the activation function
            activation = self._get_v_layer_activation(layer=l)

            # If this is the last layer
            if l == last_layer:
                # Do not activate if take only preactivation
                # i.e preactivation == True
                if not preactivation: 
                    x = activation(x)
            
            # If not last layer - activate and go to next layer
            else:
                x = activation(x)
        return x 
    
    def forward(self, x):
        return self.U(self.v(x))

def get_model(in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
    return Net(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, L=L, device=device)

# Define loss functions
def logistic_loss(y, y_positive, y_negatives):
    N, d = y.shape
    h_exp_sum = 0.0
    
    for y_negative in y_negatives:
        h_exp_sum += torch.exp(
            -torch.matmul(
                y.reshape(N, 1, d), 
                (y_positive - y_negative).reshape(N, d, 1)
            ).squeeze(1)
        )
    loss = torch.log(1 + h_exp_sum)
    return loss

# Compute Yunwen's complexity measure
def compute_complexity_YW(dataloader, network: Net, device=None):
    # Report
    print('[INFO] Computing Yunwen et. al. complexity measure...')

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary constants
    L = network.L 
    n = len(dataloader)
    d = network.out_dim

    # Initialization
    B_x = 0.0
    complexity = np.sqrt(L * d)
    
    # Compute complexity
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        complexity *= frobenius_norm(A_l)
        complexity *= spectral_norm(A_l)

    # Find B_x
    print('[INFO] Computing B_x...')
    network.eval()
    with tqdm.tqdm(total=n) as pbar:
        for i, batch in enumerate(dataloader):
            # Calculate inputs l2 norms
            X = torch.cat([batch[0], batch[1], *batch[2]], dim=0)
            X_l2 = torch.linalg.norm(X, dim=1, ord=2).squeeze()
            X_l2_max = torch.max(X_l2)

            # Check
            if B_x < X_l2_max.item():
                B_x = X_l2_max.item()

            # Update progress
            pbar.set_postfix({
                'batch' : f'#[{i+1}/{n}]',
                'current Bx' : f'{B_x:.4f}'
            })
            pbar.update(1)
    complexity = complexity * (B_x ** 2)
    complexity = complexity / np.sqrt(n)
    return complexity

# Compute our complexity measure (Thm. 1)
def compute_complexity_THM1(dataloader, network: Net, device=None):
    # Report
    print('[INFO] Computing our (THM. 1) complexity measure...')

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary data
    L = network.L
    n = len(dataloader) 

    # Initialization
    complexity = 1.0 
    B_x = 0.0

    # Compute product of spectral norms
    sum_alsl = 0.0
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        s_l = spectral_norm(A_l)
        a_l = l21_norm(A_l)
        complexity *= (s_l ** 2) 
        sum_alsl += ((a_l / s_l) ** (2/3))
    complexity *= (sum_alsl ** (3/2))

    # Compute R and B_x
    print('[INFO] Computing B_x...')
    network.eval()
    with tqdm.tqdm(total=n) as pbar:
        for i, batch in enumerate(dataloader):
            # Calculate inputs l2 norms
            X = torch.cat([batch[0], batch[1], *batch[2]], dim=0)
            X_l2 = torch.linalg.norm(X, dim=1, ord=2).squeeze()
            X_l2_max = torch.max(X_l2)

            # Check
            if B_x < X_l2_max.item():
                B_x = X_l2_max.item()

            # Update progress
            pbar.set_postfix({
                'batch' : f'#[{i+1}/{n}]',
                'current Bx' : f'{B_x:.4f}'
            })
            pbar.update(1)

    complexity = complexity * (B_x ** 2)
    complexity = complexity / np.sqrt(n)
    return complexity

# Compute our complexity measure (Thm. 2)
def compute_complexity_THM2(dataloader, network: Net, device=None):
    # Report
    print('[INFO] Computing our (THM. 2) complexity measure...')

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary data
    L = network.L
    n = len(dataloader) 

    # Initialization
    complexity = 1.0 
    B_x = 0.0
    R   = 0.0

    # Compute product of spectral norms
    sum_alsl = 0.0
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        s_l = spectral_norm(A_l)
        a_l = l21_norm(A_l)
        complexity *= s_l 
        sum_alsl += ((a_l / s_l) ** (2/3))
    complexity *= (sum_alsl ** (3/2))

    # Compute R and B_x
    print('[INFO] Computing B_x and R...')
    network.eval()
    with tqdm.tqdm(total=n) as pbar:
        for i, batch in enumerate(dataloader):
            # Calculate inputs l2 norms
            X = torch.cat([batch[0], batch[1], *batch[2]], dim=0)
            X_l2 = torch.linalg.norm(X, dim=1, ord=2).squeeze()
            X_l2_max = torch.max(X_l2)

            # Calculate network output
            y1, y2, y3 = apply_model_to_batch(network, batch, device=network.device)
            Y = torch.cat([y1, y2, *y3], dim=0)
            Y_l2 = torch.linalg.norm(Y, dim=1, ord=2).squeeze()
            Y_l2_max = torch.max(Y_l2)

            # Check
            if B_x < X_l2_max.item():
                B_x = X_l2_max.item()

            if R < Y_l2_max.item():
                R = Y_l2_max.item()

            # Update progress
            pbar.set_postfix({
                'batch' : f'#[{i+1}/{n}]',
                'current Bx' : f'{B_x:.4f}',
                'current R' : f'{R:.4f}'
            })
            pbar.update(1)

    complexity = complexity * B_x * R
    complexity = complexity / np.sqrt(n)
    return complexity

# Compute our complexity measure (Thm. 3)
def compute_complexity_THM3(dataloader, network: Net, device=None):
    # Report
    print('[INFO] Computing our (THM. 3) complexity measure...')

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary data
    L = network.L
    n = len(dataloader) 

    # Initialization
    complexity = 1.0 
    b_l = { l : 0.0 for l in range(0, L+1) } 
    a_l = { l : 0.0 for l in range(1, L+1)}
    s_l = { l : 0.0 for l in range(1, L+1)}
    rho_l = { l : 0.0 for l in range(1, L+1) }

    # Compute all a_l
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        a_l[l] = l21_norm(A_l)

    # Compute R and B_x
    print('[INFO] Computing b_L...')
    network.eval()
    with tqdm.tqdm(total=n) as pbar:
        for i, batch in enumerate(dataloader):
            # Calculate inputs l2 norms
            X = torch.cat([batch[0], batch[1], *batch[2]], dim=0)
            X_l2 = torch.linalg.norm(X, dim=1, ord=2).squeeze()
            X_l2_max = torch.max(X_l2)

            # Calculate network hidden activations
            for l in range(1, L):
                y1 = network._get_output_from_layer(batch[0].to(device), last_layer=l, preactivation=False)
                y2 = network._get_output_from_layer(batch[1].to(device), last_layer=l, preactivation=False)
                y3 = [network._get_output_from_layer(b.to(device), last_layer=l, preactivation=False) for b in batch[2]]
                Yl = torch.cat([y1, y2, *y3], dim=0)
                Yl_l2 = torch.linalg.norm(Yl, dim=1, ord=2).squeeze()
                Yl_l2_max = torch.max(Yl_l2)
                if(b_l[l] < Yl_l2_max.item()):
                    b_l[l] = Yl_l2_max.item()

            # Calculate network output
            y1, y2, y3 = apply_model_to_batch(network, batch, device=network.device)
            Y = torch.cat([y1, y2, *y3], dim=0)
            Y_l2 = torch.linalg.norm(Y, dim=1, ord=2).squeeze()
            Y_l2_max = torch.max(Y_l2)

            # Check
            if b_l[0] < X_l2_max.item():
                b_l[0] = X_l2_max.item()

            if b_l[L] < Y_l2_max:
                b_l[L] = Y_l2_max.item()

            # Update progress
            pbar.update(1)

    # Clip values of bl
    for l in range(0, L+1): b_l[l] = max(b_l[l], 1.0)

    print('[INFO] Computing rho_l...')
    for l in range(1, L+1):
        # Compute forward spectral norm products
        max_rhol = 0.0
        for u in range(l, L+1):
            forward_product = 1.0
            if(u != l):
                forward_product = np.prod([s_l[j] for j in range(l+1, u+1)])
            current_rhol = forward_product / b_l[u]
            if current_rhol > max_rhol:
                max_rhol = current_rhol
        rho_l[l] = max_rhol

    complexity = complexity * (b_l[L] ** 2)
    complexity = complexity * np.sum([(a_l[l] * b_l[l-1] * rho_l[l]) ** (2/3) for l in range(1, L+1)]) ** (3/2)
    complexity = complexity / np.sqrt(n)
    return complexity