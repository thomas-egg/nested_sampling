import numpy as np
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

'''
Test MCMC shamelessly stolen from original Diffusive
Nested Sampling Paper: https://arxiv.org/pdf/0912.2380. 
'''

def prob(level, J:int, l:float, in_bounds, max_level):
        '''
        Return joint distribution of particle position and level

        @param level : current particle level
        @param J : Current max level
        @param l : Lambda value
        '''
        return (level.level_weight(j=J, l=l, max_level=max_level) / level.get_X) * in_bounds

class MCMC(object):
    def __init__(self, beta, likelihood_function, max_J, acc_rate, iterations=10000):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.acc_rate = acc_rate
        self.likelihood_function = likelihood_function
        self.max_J = max_J
        self.iters = iterations

    def __call__(self, particle:Particle, levels, J, l):
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
            step[i] = 1
            step *= np.random.uniform(-1/S, 1/S)
            x_new = x + step
    
            # Index
            j_new = np.clip(np.round(np.random.normal(loc=j, scale=S_prime)), 0, J).astype(int)
            new_level = levels[j_new]

            # Acceptance
            in_bounds_new = self.likelihood_function(x_new) > new_level.likelihood_bound
            p_x_prime = prob(level=new_level, J=J, l=l, in_bounds=in_bounds_new, max_level=self.max_J)
            p_x = prob(level=level, J=J, l=l, in_bounds=True, max_level=self.max_J)
            a = p_x_prime / p_x
            r = min(1, a)
            u = np.random.rand()
            if u < r:
                j = j_new
                x = x_new
            else:
                j = j
                x = x

            # Accumulate points
            xs.append(x)
            js.append(j)
            likelihoods.append(self.likelihood_function(x))
        
        # Assign particle state
        particle.assign_state(new_pos=x, new_index=j)

        # Return
        return np.array(xs), np.array(js), np.array(likelihoods)