import torch
import numpy as np
import sys
import os

# Add the directory containing diffusive_nested_sampling to sys.path
sys.path.append(os.path.abspath( "../.."))

# Now import the modules
from diffusive_nested_sampling.dns import DiffusiveNestedSampler
from diffusive_nested_sampling.mc import MCMC

# Likelihood function
def likelihood(x):
    '''
    20-D spike and slab likelihood function

    @param x : coordinate
    @return L : likelihood
    '''
    u = 0.01
    v = 0.1

    # Spike
    t1 = np.sum(-0.5 * (x / v) ** 2) - x.size * np.log(v * np.sqrt(2 * np.pi))
    
    # Slab
    t2 = np.sum(-0.5 * ((x - 0.031) / u) ** 2) - x.size * np.log(u * np.sqrt(2 * np.pi))
    
    return (np.exp(t1) + (100 * np.exp(t2)))

# Instantiate sampler
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = MCMC(beta=0, likelihood_function=likelihood, max_J=100, acc_rate=0.25)
    dns = DiffusiveNestedSampler(likelihood, n_particles=1, dim=20, max_level=100, sampler=sampler, device=device)

    # Run sampler
    chain, levels = dns(1100000)
    return chain, levels