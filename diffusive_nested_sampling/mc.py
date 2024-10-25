import torch
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

def prob(level, J:int, l:float):
        '''
        Return joint distribution of particle position and level

        @param level : current particle level
        @param J : Current max level
        @param l : Lambda value
        '''
        return level.level_weight(j=J, l=l) / level.X

class MCMC(object):
    def __init__(self, beta, likelihood_function, acc_rate=0.5):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.acc_rate = acc_rate
        self.likelihood_function = likelihood_function

    def __call__(self, particle:Particle, levels, J, l):
        '''
        Run sampling iteration

        @param particle : particle
        @param levels : list of levels
        @param J : current highest index
        @param l : lambda
        '''
        
        # Set up proposal
        # Position
        j, x = particle.j, particle.pos
        level = levels[j]
        i = torch.randint(size=(1,), low=0, high=x.shape[-1]).item()
        step = torch.zeros(x.shape)
        step[i] = torch.rand(size=(1,)).item() - 0.5
        x_new = x + step
        if self.likelihood_function(x_new) > levels[j].likelihood_bound:
            x = x_new
        else:
            x = x

        # Index
        if J > 1:
            j_new = (j + (2 * torch.randint(0, 2, (1,))) - 1).clamp(min=0, max=J-1)
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
        else:
            j = j

        # Return
        return x, j