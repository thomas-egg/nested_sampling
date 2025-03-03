import torch
import numpy as np
from diffusive_nested_sampling.level import Level
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
    def __init__(self, likelihood_func, n_particles, dim, max_level, sampler, device='cpu'):
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
        self.levels = [Level(0, 0.0, prev_X=None)]
        self.max_level = max_level
        self.sampler = sampler
        self.sampler.iters = int(self.sampler.iters / self.n)
        self.device = device

        # Initialize particles
        pos = np.random.uniform(low=-0.5, high=0.5, size=dim)
        self.p = [Particle(pos, 0) for _ in range(n_particles)]
        self.chain = {
            'x' : [p.pos for p in self.p],
            'j' : [p.j for p in self.p],
            'L' : [self.likelihood_func(particle.pos) for particle in self.p]
        }
        self.counter = np.zeros(self.max_level)

    def run_mcmc(self, particle, levels, J, L, chain_length):
        '''
        Run MCMC for particle

        @param particle : particle to run MCMC for
        @param levels : levels
        @param J : number of levels
        @param L : lambda value
        '''
        p, x, j, l = self.sampler(particle, levels, J, L, chain_length)
        return p, x, j, l

    def __call__(self, nsteps, L=10, C=1000):
        '''
        Call DNS

        @param iter_per_level : likelihood evaluations per level created
        @param L : lambda value
        '''
        J = 0
        all_js = np.array([])
        for i in tqdm(range(int(nsteps / (self.n * self.sampler.iters)))):

            # Run MC here
            with Pool(self.n) as pool:
                results = pool.starmap(self.run_mcmc, [(p, self.levels, J, L, len(self.chain['j'])) for p in self.p])
                new_p, new_x, new_j, new_L = zip(*results)
                self.p = new_p
                new_x = np.concatenate(new_x, axis=0)
                new_j = np.concatenate(new_j, axis=0)
                new_L = np.concatenate(new_L, axis=0)                
                self.chain['x'] = np.concatenate((self.chain['x'], new_x), axis=0)
                self.chain['j'] = np.concatenate((self.chain['j'], new_j), axis=0)
                self.chain['L'] = np.concatenate((self.chain['L'], new_L), axis=0)
                all_js = np.concatenate((all_js, new_j), axis=0)
                filtered_new_j = new_j[new_j < J]
                self.counter += np.bincount(filtered_new_j, minlength=len(self.counter))

            if J < self.max_level:

                # Add level
                J += 1
                likelihoods = torch.tensor(self.chain['L']).to(self.device)
                boundary = torch.quantile(likelihoods, q=(1 - np.exp(-1))).item()
                self.levels.append(Level(index=J, likelihood_boundary=boundary, prev_X=self.levels[J-1].get_X))

                # Remove points lower than new boundary
                inds = likelihoods > boundary
                inds = inds.cpu().numpy()
                self.chain['x'] = np.array(self.chain['x'])[inds]
                self.chain['j'] = np.array(self.chain['j'])[inds]
                self.chain['L'] = np.array(self.chain['L'])[inds]

            # Adjust level weights
            for j in range(J):
                self.levels[j].set_visits(np.sum(all_js == j))
                if j > 1:
                    self.levels[j].set_X(self.levels[j-1].get_X, self.chain, self.counter, C)

        return self.chain, self.levels, all_js