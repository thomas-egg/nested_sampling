import torch
from particle import Particle
from level import Level

def prob(self, X:float, likelihood:float, J:int, l:float):
        '''
        Return joint distribution of particle position and level

        @param level : current particle level
        @param X : Phase space volume
        @param J : Current max level
        @param l : Lambda value
        '''
        if likelihood < self.level().likelihood_bound():
            return 0
        else:
            return self.level().level_weight(J, l) / X

class MCMC(object):
    def __init__(self, beta, C, likelihood_function, acc_rate=0.8):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.C = C
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
        j, x = particle.j(), particle.pos()
        level = levels[j]
        x_new = x + torch.rand_like(x)
        j_new = j + (2 * torch.randint(0, 2, (1,))) - 1
        new_level = levels[j_new]

        # Acceptance
        p_x_prime = prob(level.X, self.likelihood_function(x_new), J, l)
        p_x = prob(new_level.X, self.likelihood(x), J, l)
        a = p_x_prime / p_x
        r = torch.min(1, a)
        if torch.rand() < r:
            return x_new, j_new
        else:
            return x, j