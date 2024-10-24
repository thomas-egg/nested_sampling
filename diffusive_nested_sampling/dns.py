import torch
import numpy as np
from diffusive_nested_sampling.level import Level
from diffusive_nested_sampling.particle import Particle

'''
Diffusive Nested Sampling Code

Implementation of Diffusive Nested Sampling - introduced by 
Brewer et al. (https://arxiv.org/pdf/0912.2380), this method
should result in a significant speedup of the sampling of
the normalizing constant (partition function)
'''

class DiffusiveNestedSampler(object):
    def __init__(self, likelihood_func, n_particles, dim, max_level, sampler):
        '''
        Initialize sampler

        @param likelihood_func : likelihood function
        @param n_particles : number of particles for sampler
        @param L : lambda value for backtracking control
        @param dim : dimensionality of system
        @param max_level : maximum number of levels
        '''
        self.likelihood_func = likelihood_func
        self.n = n_particles
        self.levels = [Level(index=0, likelihood_boundary=0)]
        self.max_level = max_level
        self.sampler = sampler

        # Initialize particles
        pos = torch.zeros(dim)
        self.p = Particle(pos, 0)
        self.likelihoods = [self.likelihood_func(pos)]

    def __call__(self, iter_per_level=1000, L=10, C=1000):
        '''
        Call DNS

        @param iter_per_level : likelihood evaluations per level created
        @param L : lambda value
        '''
        J = 1
        while J < self.max_level:
            for i in range(iter_per_level):

                # Run MC here
                x, j = self.sampler(self.p, self.levels, J, L)
                self.p.assign_state(x, j)
                self.likelihoods.append(self.likelihood_func(x))

            # Adjust level weights
            self.likelihoods = torch.tensor(self.likelihoods)
            print(self.likelihoods)
            for i in range(J):
                self.levels[i].set_X(self.levels[i-1].get_X, self.likelihoods, self.p.history)
            
            # Add new level
            boundary = torch.quantile(self.likelihoods, q=(1 - np.exp(-1)))
            print(boundary)
            self.likelihoods = self.likelihoods[self.likelihoods > boundary]
            self.levels.append(Level(index=J, likelihood_boundary=boundary))
            J += 1
            print(f'Added LEVEL {J}')

            # Remove likelihoods lower than new boundary
            self.likelihoods = self.likelihoods[self.likelihoods > boundary].tolist()

        # Return likelihoods afterwards
        return torch.tensor(self.likelihoods)