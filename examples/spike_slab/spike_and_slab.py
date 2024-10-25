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
def likelihood(x:torch.tensor):
    '''
    20-D spike and slab likelihood function

    @param x : coordinate
    @return L : likelihood
    '''
    u = 0.01
    v = 0.1

    # Spike
    t1 = torch.sum(-0.5 * (x / v) ** 2) - x.size(0) * torch.log(v * torch.sqrt(torch.tensor(2) * torch.pi))
    
    # Slab
    t2 = torch.sum(-0.5 * ((x - 0.031) / u) ** 2) - x.size(0) * torch.log(u * torch.sqrt(torch.tensor(2) * torch.pi))
    
    return torch.exp(t1) + (100 * torch.exp(t2))

# Instantiate sampler
def main():
    sampler = MCMC(beta=0, likelihood_function=likelihood)
    dns = DiffusiveNestedSampler(likelihood, n_particles=1, dim=20, max_level=100, sampler=sampler)

    # Run sampler
    likelihoods, p = dns(10000)
    return likelihoods, p