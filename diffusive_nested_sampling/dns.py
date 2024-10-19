import torch
from level import Level
from particle import Particle

'''
Diffusive Nested Sampling Code

Implementation of Diffusive Nested Sampling - introduced by 
Brewer et al. (https://arxiv.org/pdf/0912.2380), this method
should result in a significant speedup of the sampling of
the normalizing constant (partition function)
'''

class DiffusiveNestedSampler(object):
    def __init__(self, likelihood_func, n_particles, L, dim, max_level, beta, C, sampler):
        '''
        Initialize sampler

        @param likelihood_func : likelihood function
        @param n_particles : number of particles for sampler
        @param L : lambda value for backtracking control
        @param dim : dimensionality of system
        @param max_level : maximum number of levels
        @param beta : term for weighing acceptance probability
        @param C : confidence
        '''
        self.likelihood_func = likelihood_func
        self.n = n_particles
        self.L = L
        self.levels = [Level(index=0, likelihood_boundary=torch.finfo(torch.float32).min)]
        self.max_level = max_level
        self.beta = beta
        self.sampler = sampler

        # Initialize particles
        pos = torch.rand(dim)
        self.p = Particle(pos, 0)
        self.likelihoods = [self.likelihood_func(pos)]

    def __call__(self, iter_per_level=10000):
        '''
        Call DNS

        @param iter_per_level : likelihood evaluations per level created
        '''
        J = 1
        while J < self.max_level:
            for i in range(iter_per_level):

                # Run MC here
                x, j = self.sampler(self.p, self.levels, J, self.L)
                self.p.assign_state(x, j)

            # Add new level
            boundary = torch.quantile(self.likelihoods)
            J += 1
            self.levels.append(Level(index=J, likelihood_boundary=boundary))