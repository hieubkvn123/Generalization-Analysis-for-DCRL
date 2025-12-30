import torch
import torch.nn as nn

import tqdm
import numpy as np
from common import get_default_device
from norms import frobenius_norm, lp_norm, l21_norm, spectral_norm 

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
        x = x.view(x.size(0), -1)
        return self.U(self.v(x))

def get_model(in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
    return Net(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, L=L, device=device)

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

# Compute Bartlett et al. complexity
def compute_complexity_bartlett(network: Net, n=1000, device=None):
    # Report
    print('[INFO] Computing Bartlett et al. complexity measure...')
    network.eval()

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary constants
    L = network.L 
    d = network.out_dim

    # Initialization
    R_A = 0.0
    
    # Compute complexity
    prod_term, sum_term = 1.0, 0.0
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)

        # Compute all necessary norms
        s_l = spectral_norm(A_l)
        a_l = l21_norm(A_l)

        # Compute the sum and product terms
        prod_term *= s_l
        sum_term  += (a_l / s_l) ** (2/3)

    # Compute spectral complexity
    R_A = prod_term * (sum_term ** (3/2))

    # Scale by 1/sqrt(n)
    complexity = R_A / np.sqrt(n)
    return complexity

# Compute our complexity 
def compute_complexity_ours(network: Net, n=1000, p=0.5, device=None):
    # Report
    print('[INFO] Computing our complexity measure...')
    network.eval()

    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Get necessary constants
    L = network.L 
    d = network.out_dim

    # Initialization
    R_A = 0.0

    # Compute the product term
    prod_term = 1.0
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        s_l = spectral_norm(A_l)
        prod_term += s_l
    
    # Compute complexity
    R_A = 0.0
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        d_out, d_in = A_l.shape

        # Compute all necessary norms
        s_l = spectral_norm(A_l)
        m_l = lp_norm(A_l, p=p)

        # Compute U_l
        if l != L:
            U_l = (m_l / s_l) * np.sqrt((d_out ** 2) * d_in) 
        else:
            U_l = (m_l / s_l) * np.sqrt(d_out * d_in) 
        R_A += (U_l*prod_term) ** ((2*p) / (3*p + 2))
    complexity = R_A ** ((3*p + 2)/(2*p + 4))

    # Scale by 1/sqrt(n)
    complexity = complexity / np.sqrt(n)
    return complexity
