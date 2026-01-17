import torch
import torch.nn as nn

import tqdm
import itertools
import numpy as np
from common import get_default_device
from norms import frobenius_norm, l0_norm, lp_norm, l21_norm, spectral_norm 

# Network definition
class ReLUDropout(nn.Module):
    def __init__(self, rate=0.2):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(rate)
    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Net(nn.Module):
    def __init__(self, in_dim=784, out_dim=64, hidden_dim=128, spec_norm=False, L=10, device=None):
        super().__init__()
        
        # Store configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.spec_norm = spec_norm
        self.L = L

        # For re-loading model
        self._init_args = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_dim': hidden_dim,
            'L': L
        }

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
        if not self.spec_norm:
            for _ in range(1, self.L):
                self.fc_hidden_layers.append( nn.Linear(hidden_dim, hidden_dim, bias=False) )
                self.fc_hidden_layers.append( nn.ReLU() )
            self.v = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.ReLU(),
                *self.fc_hidden_layers
            )
            self.U = nn.Linear(hidden_dim, out_dim)
        else:
            print('[INFO] Spectral normalization is applied...')
            for _ in range(1, self.L):
                self.fc_hidden_layers.append( 
                    nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))
                )
                self.fc_hidden_layers.append( nn.ReLU() )
            self.v = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim, bias=False)),
                nn.ReLU(),
                *self.fc_hidden_layers
            )
            self.U = nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

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

    def get_weight_matrices(self):
        weight_matrices = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight_matrices.append(module.weight)
        return weight_matrices
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.U(self.v(x))

def get_model(in_dim=784, out_dim=64, hidden_dim=128, spec_norm=False, L=10, device=None):
    return Net(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, spec_norm=spec_norm, L=L, device=device)

def prune_matrix(matrix, threshold=1e-3):
    pruned = matrix.copy()
    pruned[np.abs(pruned) < threshold] = 0
    return pruned

## Save and load functions ##
def save_model(model, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__,
        'model_init_args': getattr(model, '_init_args', {}),
    }, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    # Automatically detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filename, map_location=device)

    # Get model class and initialization arguments
    model_class = checkpoint['model_class']
    init_args = checkpoint.get('model_init_args', {})

    # Recreate model architecture
    if init_args:
        model = model_class(**init_args)
    else:
        model = model_class()

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from {filename} on device: {device}")
    return model


### Complexity Computation ###
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
        A_l = prune_matrix(A_l)

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

# Compute Para. count complexity 
def compute_complexity_paracount(network: Net, n=1000, device=None):
    # Report
    print('[INFO] Computing Graf et al. (para-count) complexity measure...')
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
    W = 0.0
    
    # Compute complexity
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        d_out, d_in = A_l.shape
        W += d_out * d_in

    # Scale by 1/sqrt(n)
    complexity = np.sqrt((L * W)/n)
    return complexity

# Compute Para. count complexity 
def compute_complexity_paracount(network: Net, n=1000, device=None):
    # Report
    print('[INFO] Computing Graf et al. (para-count) complexity measure...')
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
    W = 0.0
    
    # Compute complexity
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        d_out, d_in = A_l.shape
        W += d_out * d_in

    # Scale by 1/sqrt(n)
    complexity = np.sqrt((L * W)/n)
    return complexity

# Compute non-zero Para. count complexity 
def compute_complexity_paracount_nonzero(network: Net, n=1000, device=None):
    # Report
    print('[INFO] Computing Graf et al. (para-count) complexity measure...')
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
    W = 0.0
    
    # Compute complexity
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        A_l = prune_matrix(A_l)
        W += l0_norm(A_l)

    # Scale by 1/sqrt(n)
    complexity = np.sqrt((L * W)/n)
    return complexity

# Compute our complexity 
def compute_complexity_ours(network: Net, n=1000, p=0.1, device=None, verbose=True):
    # Report
    if verbose: print('[INFO] Computing our complexity measure...')
    network.eval()

    # Get the list of orders
    if isinstance(p, float):
        p = [p] * network.L
    assert len(p) == network.L

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
        A_l = prune_matrix(A_l)
        s_l = spectral_norm(A_l)
        prod_term += s_l
    
    # Compute complexity
    R_A = 0.0
    layers = list(range(1, L+1))
    for p_l, l in zip(p, layers):
        A_l = network._get_v_layer_weights(layer=l)
        A_l = prune_matrix(A_l)
        d_out, d_in = A_l.shape

        # Compute all necessary norms
        s_l = spectral_norm(A_l)
        m_l = np.sum(np.abs(A_l) ** p_l)
        W_l = np.sqrt((d_out ** 2) * d_in)
        if l == L: W_l = np.sqrt(d_out * d_in)

        # Compute U_l
        U_l = m_l ** (2/(3*p_l + 2)) * ( (W_l * (prod_term / s_l)) ** ((2*p_l) / (3*p_l + 2)) )
        R_A += U_l 
    rho = np.max(p)
    complexity = R_A ** ((3*rho + 2)/(2*rho + 4))
    complexity = complexity * np.sqrt(L)

    # Scale by 1/sqrt(n)
    complexity = complexity / np.sqrt(n)
    return complexity

# Compute our complexity with layer-wise optimal p
def compute_complexity_ours_opt(network: Net, n=1000, device=None):
    print('[INFO] Computing our complexity measure with layer-wise optimal p...')
    
    # Get device
    if device is None:
        device = get_default_device()
        network = network.to(device)
        network.device = device

    # Define p search space
    L = network.L
    p_candidates = np.arange(0.05, 1.0, 0.05)
    
    # Grid search over all combinations of p values
    best_complexity = float('inf')
    best_p_values = None
    
    for p_combination in itertools.product(p_candidates, repeat=L):
        complexity = compute_complexity_ours(network, n=n, p=list(p_combination), device=device, verbose=False)
        if complexity < best_complexity:
            best_complexity = complexity
            best_p_values = p_combination
    
    return best_complexity
