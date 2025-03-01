import numpy as np
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

'''
Test MCMC shamelessly stolen from original Diffusive
Nested Sampling Paper: https://arxiv.org/pdf/0912.2380. 
'''

class MCMC(object):
    def __init__(self, beta, likelihood_function, max_J, acc_rate, iterations=10000, C=1000):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.acc_rate = acc_rate
        self.likelihood_function = likelihood_function
        self.max_J = max_J - 1
        self.iters = iterations
        self.C = C

    def __call__(self, particle:Particle, levels, J, l, chain_length):
        '''
        Run sampling iteration

        @param particle : particle
        @param levels : list of levels
        @param J : current highest index
        @param l : lambda
        '''
        
        xs = []
        js = []
        likelihoods = []
        j, x = particle.j, particle.pos
        for _ in range(self.iters):

            # Set up proposal - Jeffreys Prior
            S = 10**np.random.uniform(np.log10(10**-6), np.log10(1))
            S_prime = 10**np.random.uniform(np.log10(1), np.log10(100))

            # Position
            level = levels[j]
            i = np.random.randint(low=0, high=x.shape[-1])
            step = np.zeros(x.shape)
            step[i] = 1.0
            step *= np.random.uniform(-1/S, 1/S)
            x_new = x + step
    
            # Index
            j_new = np.round(np.random.normal(loc=j, scale=S_prime)).astype(int)
            if j_new > J or j_new < 0:
                xs.append(x)
                js.append(j)
                likelihoods.append(self.likelihood_function(x))
                continue

            # Update level
            new_level = levels[j_new]
            j_weight, j_visits, j_exp_visits = level.level_weight(j=J, l=l, max_level=self.max_J, chain_length=chain_length)
            j_prime_weight, j_prime_visits, j_prime_exp_visits = new_level.level_weight(j=J, l=l, max_level=self.max_J, chain_length=chain_length)
            a = (j_prime_weight / j_weight) * (((j_visits + self.C) / (j_exp_visits + self.C)) / ((j_prime_visits + self.C) / (j_prime_exp_visits + self.C))) ** self.beta
            r = min(1, a)
            u = np.random.rand()
            if u < r:
                j = j_new
            else:
                j = j

            if self.likelihood_function(x_new) > levels[j].likelihood_bound:
                x = x_new
            else:
                x = x

            # Accumulate points
            xs.append(x)
            js.append(j)
            likelihoods.append(self.likelihood_function(x))
        
        # Assign particle state
        particle.assign_state(new_pos=x, new_index=j)

        # Return
        return particle, np.array(xs), np.array(js), np.array(likelihoods)