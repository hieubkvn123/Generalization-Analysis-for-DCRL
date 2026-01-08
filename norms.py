import numpy as np

# Some utility
def l21_norm(A):
    norm = 0.0
    for j in range(A.shape[1]):
        aj_l2 = np.linalg.norm(A[:, j])
        norm += aj_l2
    return norm

def frobenius_norm(A):
    return np.linalg.norm(A, ord='fro')

def spectral_norm(A):
    sv = np.linalg.svd(A).S
    return max(sv)

def lp_norm(A, p=0.5):
    return np.sum(np.abs(A) ** p) ** (1.0 / p)

def l0_norm(A):
    return np.count_nonzero(A)
