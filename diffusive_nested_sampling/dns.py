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
    def __init__(self, log_likelihood_func, n_particles, dim, max_level, sampler, device='cpu'):
        '''
        Initialize sampler

        @param log_likelihood_func : log_likelihood function
        @param n_particles : number of particles for sampler
        @param L : lambda value for backtracking control
        @param dim : dimensionality of system
        @param max_level : maximum number of levels
        '''
        self.log_likelihood_func = log_likelihood_func
        self.n = n_particles
        self.levels = [Level(0, np.log(0.0), prev=None)]
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
            'L' : [self.log_likelihood_func(particle.pos) for particle in self.p]
        }
        self.counter = np.zeros(self.max_level)

    def run_mcmc(self, particle, levels, J, L):
        '''
        Run MCMC for particle

        @param particle : particle to run MCMC for
        @param levels : levels
        @param J : number of levels
        @param L : lambda value
        '''
        p, x, j, l, t, a, e = self.sampler(particle, levels, J, L)
        return p, x, j, l, t, a, e

    def __call__(self, nsteps, L=10.0, C=1000):
        '''
        Call DNS

        @param iter_per_level : log_likelihood evaluations per level created
        @param L : lambda value
        '''
        J = 0
        all_js = np.array([])
        for i in tqdm(range(int(nsteps / (self.n * self.sampler.iters)))):

            # Run MC here
            with Pool(self.n) as pool:
                results = pool.starmap(self.run_mcmc, [(p, self.levels, J, L) for p in self.p])
                new_p, new_x, new_j, new_L, new_t, new_a, new_e = zip(*results)
                self.p = new_p
                new_x = np.concatenate(new_x, axis=0)
                new_j = np.concatenate(new_j, axis=0)
                new_L = np.concatenate(new_L, axis=0)
                sum_t = np.sum(new_t, axis=0)
                sum_a = np.sum(new_a, axis=0)
                sum_e = np.sum(new_e, axis=0)
                self.chain['x'] = np.concatenate((self.chain['x'], new_x), axis=0)
                self.chain['j'] = np.concatenate((self.chain['j'], new_j), axis=0)
                self.chain['L'] = np.concatenate((self.chain['L'], new_L), axis=0)
                all_js = np.concatenate((all_js, new_j), axis=0)

            # Adjust level weights
            for j in range(0, len(self.levels)):
                self.levels[j].set_visits(total=sum_t[j], x_adj=sum_a[j], exceeds=sum_e[j])
                if j > 1:
                    self.levels[j].set_log_X(self.levels[j-1].get_log_X, C)

            if J < self.max_level:

                # Add level
                J += 1
                log_likelihoods = torch.tensor(self.chain['L'])#.to(self.device)
                boundary = torch.quantile(log_likelihoods, q=(1 - np.exp(-1))).item()
                self.levels.append(Level(index=J, log_likelihood_boundary=boundary, prev=self.levels[J-1].get_log_X))

                # Remove points lower than new boundary
                inds = log_likelihoods > boundary
                inds = inds.cpu().numpy()
                self.chain['L'] = np.array(self.chain['L'])[inds]

        return self.chain, self.levels, all_js