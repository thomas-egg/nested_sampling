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
    max_level = 100
    sampler = MCMC(beta=10, log_likelihood_function=log_likelihood, max_J=max_level, iterations=10000)
    dns = DiffusiveNestedSampler(n_particles=1, dim=20, max_level=max_level, sampler=sampler, L=10.0)

    # Run sampler
    chain, levels, particles = dns(nsteps=10000000)
    return chain, levels, particles