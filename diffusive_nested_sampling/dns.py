import torch
import numpy as np
from diffusive_nested_sampling.level import Level
from diffusive_nested_sampling.levels import Levels
from diffusive_nested_sampling.particle import Particle
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

'''
Diffusive Nested Sampling Code

Implementation of Diffusive Nested Sampling - introduced by 
Brewer et al. (https://arxiv.org/pdf/0912.2380), this method
should result in a significant speedup of the sampling of
the normalizing constant (partition function)
'''

set_start_method('spawn', force=True)
class DiffusiveNestedSampler(object):
    def __init__(self, n_particles, dim, max_level, sampler, L=10.0, C=1000):
        '''
        Initialize sampler

        @param log_likelihood_func : log_likelihood function
        @param n_particles : number of particles for sampler
        @param L : lambda value for backtracking control
        @param dim : dimensionality of system
        @param max_level : maximum number of levels
        '''
        self.n = n_particles
        self.max_level = max_level
        self.sampler = sampler
        self.sampler.iters = int(self.sampler.iters / self.n)
        self.levels = Levels(self.max_level, L, C)

        # Initialize particles
        pos = np.random.uniform(low=-0.5, high=0.5, size=dim)
        self.p = [Particle(pos, 0) for _ in range(n_particles)]
        self.chain = {
            'x' : [p.pos for p in self.p],
            'j' : [p.j for p in self.p],
            'L' : [-float('inf') for particle in self.p]
        }

    def run_mcmc(self, particle, levels):
        '''
        Run MCMC for particle

        @param particle : particle to run MCMC for
        @param levels : levels
        @param J : number of levels
        @param L : lambda value
        '''
        p, x, j, l, visits, xadj, exceeds = self.sampler(particle, levels)
        return p, x, j, l, visits, xadj, exceeds

    def __call__(self, nsteps):
        '''
        Call DNS

        @param iter_per_level : log_likelihood evaluations per level created
        @param L : lambda value
        '''
        J = 0
        all_js = np.array([])
        for _ in tqdm(range(int(nsteps / (self.n * self.sampler.iters)))):

            # Run MC here
            with Pool(self.n) as pool:
                results = pool.starmap(self.run_mcmc, [(p, self.levels) for p in self.p])
                new_p, new_x, new_j, new_L, new_visits, new_xadj, new_exceeds = zip(*results)
                self.p = new_p
                new_x = np.concatenate(new_x, axis=0)
                new_j = np.concatenate(new_j, axis=0)
                new_L = np.concatenate(new_L, axis=0)
                self.chain['x'] = np.concatenate((self.chain['x'], new_x), axis=0)
                self.chain['j'] = np.concatenate((self.chain['j'], new_j), axis=0)
                self.chain['L'] = np.concatenate((self.chain['L'], new_L), axis=0)
                all_js = np.concatenate((all_js, new_j), axis=0)
                new_visits = np.sum(new_visits, axis=0)
                new_xadj = np.sum(new_xadj, axis=0)
                new_exceeds = np.sum(new_exceeds, axis=0)

            # Append
            if J < self.max_level:

                # Add level
                boundary = np.quantile(new_L, q=(1 - np.exp(-1)))
                print(boundary)
                self.levels.append(boundary=boundary)

                # Remove points lower than new boundary
                inds = self.chain['L'] >= boundary
                self.chain['L'] = self.chain['L'][inds]
                J += 1

            # Update levels
            self.levels.update_levels(new_visits, new_xadj, new_exceeds)

        return self.chain, self.levels, all_js