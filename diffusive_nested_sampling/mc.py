import torch
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

'''
Test MCMC shamelessly stolen from original Diffusive
Nested Sampling Paper: https://arxiv.org/pdf/0912.2380. 
'''

def prob(level, J:int, l:float):
        '''
        Return joint distribution of particle position and level

        @param level : current particle level
        @param J : Current max level
        @param l : Lambda value
        '''
        return level.level_weight(j=J, l=l) / level.X

class MCMC(object):
    def __init__(self, beta, likelihood_function, max_J, acc_rate=0.5):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.acc_rate = acc_rate
        self.likelihood_function = likelihood_function
        self.max_J = max_J

    def __call__(self, particle:Particle, levels, J, l):
        '''
        Run sampling iteration

        @param particle : particle
        @param levels : list of levels
        @param J : current highest index
        @param l : lambda
        '''
        
        # Set up proposal - Jeffreys Prior
        S = (1e-6 - 1) * torch.rand() + 1
        S_prime = (1 - 100) * torch.rand() + 100

        # Position
        j, x = particle.j, particle.pos
        level = levels[j]
        i = torch.randint(size=(1,), low=0, high=x.shape[-1]).item()
        step = torch.zeros(x.shape)
        step[i] = 1 / S
        x_new = x + step
        if self.likelihood_function(x_new) > levels[j].likelihood_bound:
            x = x_new
        else:
            x = x

        # Index
        if J > 1:
            j_new = torch.round(torch.normal(mean=j, std=S_prime)).clamp(min=0, max=J)
        else:
            j_new = j
        new_level = levels[j_new]

        # Acceptance
        p_x_prime = prob(new_level, J, l)
        p_x = prob(level, J, l)
        a = p_x_prime / p_x
        r = min(1, a)
        u = torch.rand(size=(1,))
        if u < r or u > self.acc_rate:
            j = j_new
        elif u < 0.5 and J == self.max_J:
            j = j_new
        else:
            j = j

        # Return
        return x, j