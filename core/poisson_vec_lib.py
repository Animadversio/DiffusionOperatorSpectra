import torch
import torch.nn.functional as F
import scipy.special as sp
import numpy as np
import math
# Define the function S_d for a given dimension d
def S_d(d):
    return (2 * np.pi**(d/2)) / sp.gamma(d/2)


def compute_E_2d(r, sigma=1, Q=1):
    # Compute the error function and exponential term
    erf_term = torch.erf(r / (math.sqrt(2) * sigma))
    exp_term = torch.exp(-r**2 / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
    # Compute the electric field E(r)
    E_r = (Q / (2 * torch.pi * r)) * (erf_term - r * exp_term)
    return E_r

def compute_Phi_2d(r, sigma=1, Q=1, r0=1):
    erf_term = torch.erf(r / (math.sqrt(2) * sigma))
    Phi_r = (Q / (2 * torch.pi)) * torch.log(r / r0) * erf_term
    return Phi_r


def compute_E_ndim(r, ndim, sigma=1, Q=1):
    assert ndim > 2 and isinstance(ndim, int)
    # Compute the error function and exponential term
    erf_term = torch.erf(r / (math.sqrt(2) * sigma))
    exp_term = torch.exp(-r**2 / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
    # Compute the electric field E(r)
    # (d-2)S_{d}\epsilon_{d}r^{d-1}
    E_r = (Q / ((ndim - 2) * S_d(ndim) * r ** (ndim - 1))) * (erf_term - r * exp_term)
    return E_r
    

def compute_Phi_ndim(r, ndim, sigma=1, Q=1, r0=1):
    assert ndim > 2 and isinstance(ndim, int)
    erf_term = torch.erf(r / (math.sqrt(2) * sigma))
    Phi_r = (Q / ((ndim - 2) * S_d(ndim) * r ** (ndim - 2))) * erf_term
    return Phi_r


def compute_E_vec_field(pnts, mu, sigma=1, Q=1, ndim=2, eps=1e-3):
    # Compute the distance from the origin
    vec = pnts - mu
    r = torch.norm(vec, dim=1)
    unit_vec = vec / r[:, None]
    r_reg = torch.sqrt(r**2 + eps**2)
    # Compute the electric field E(r)
    if ndim == 2:
        E_r = compute_E_2d(r_reg, sigma, Q)
    else:
        E_r = compute_E_ndim(r_reg, ndim, sigma, Q)
    # Compute the electric field vector field
    E_vec = - E_r[:, None] * unit_vec
    return E_vec


def compute_Phi_potential_field(pnts, mu, sigma=1, Q=1, ndim=2, eps=1e-3):
    # Compute the distance from the origin
    vec = pnts - mu
    r = torch.norm(vec, dim=1)
    r_reg = torch.sqrt(r**2 + eps**2)
    # Compute the electric field E(r)
    if ndim == 2:
        Phi_r = compute_Phi_2d(r_reg, sigma, Q)
    else:
        Phi_r = compute_Phi_ndim(r_reg, ndim, sigma, Q)
    return Phi_r
    

def compute_E_vec_field_multi_mu_split(pnts, mus, sigma=1, Q=1, ndim=2, eps=1e-3):
    vec_col = [compute_E_vec_field(pnts, mu, sigma, Q, ndim, eps) for mu in mus]
    vec_split = torch.stack(vec_col, dim=1)
    vec_total = torch.sum(vec_split, dim=1)
    return vec_total, vec_split


def compute_Phi_potential_field_multi_mu_split(pnts, mus, sigma=1, Q=1, ndim=2, eps=1e-3):
    Phi_col = [compute_Phi_potential_field(pnts, mu, sigma, Q, ndim, eps) for mu in mus]
    Phi_split = torch.stack(Phi_col, dim=1)
    Phi_total = torch.sum(Phi_split, dim=1)
    return Phi_total, Phi_split
    
    
# Example usage
# pnts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Example points
# mu = torch.tensor([0.0, 0.0, 0.0]) 
