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
def log_likelihood(x):
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
    t2 = np.sum(-0.5 * ((x - 0.031) / u) ** 2) - x.size * np.log(u * np.sqrt(2 * np.pi)) + np.log(100.0)

    logL = np.logaddexp(t1, t2)
    return logL

# Instantiate sampler
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = MCMC(beta=10, log_likelihood_function=log_likelihood, max_J=100, acc_rate=0.25, iterations=10000)
    dns = DiffusiveNestedSampler(log_likelihood, n_particles=5, dim=20, max_level=100, sampler=sampler, device=device)

    # Run sampler
    chain, levels, js = dns(nsteps=2000000, L=10.0)
    return chain, levels, js