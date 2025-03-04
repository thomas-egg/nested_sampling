import numpy as np
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

'''
Test MCMC shamelessly stolen from original Diffusive
Nested Sampling Paper: https://arxiv.org/pdf/0912.2380. 
'''

class MCMC(object):
    def __init__(self, beta, log_likelihood_function, max_J, acc_rate, iterations=10000, C=1000):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.acc_rate = acc_rate
        self.log_likelihood_function = log_likelihood_function
        self.max_J = max_J
        self.iters = iterations
        self.C = C

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
        log_likelihoods = []
        total_visits = np.zeros(J+1)
        visits_x_adj = np.zeros(J+1)
        exceeds = np.zeros(J+1)
        j, x = particle.j, particle.pos
        for _ in range(self.iters):

            # Set up proposal - Jeffreys Prior
            S = 10**np.random.uniform(np.log10(10**-6), np.log10(1))
            S_prime = 10**np.random.uniform(np.log10(1), np.log10(100))

            # Position
            level = levels[j]
            i = np.random.randint(x.shape[-1])
            step = np.zeros_like(x)
            step[i] = np.random.uniform(-1/S, 1/S)
            x_new = np.clip(x + step, -0.5, 0.5)

            # Update x
            new_logL = self.log_likelihood_function(x_new)
            if new_logL > level.log_likelihood_bound:
                x = x_new
            else:
                new_logL = self.log_likelihood_function(x)
                x = x

            # Update level
            j_new = int(np.clip(np.random.normal(loc=j, scale=S_prime), 0, J))
            new_level = levels[j_new]
            j_log_weight = level.level_weight(j=J, l=l, max_level=self.max_J)
            j_prime_log_weight = new_level.level_weight(j=J, l=l, max_level=self.max_J)
            a = (new_logL > new_level.log_likelihood_bound) * np.exp(j_prime_log_weight - j_log_weight) 
            r = min(1, a)
            u = np.random.rand()
            if u < r:
                j = j_new
            else:
                j = j

            # Accumulate quantities
            if j < J:
                visits_x_adj[j] += 1
                if new_logL > levels[j+1].log_likelihood_bound:
                    exceeds[j] += 1
            total_visits[j] += 1

            # Thin chain
            if i % 1000 == 0:
                xs.append(x)
                js.append(j)
                log_likelihoods.append(self.log_likelihood_function(x))
        
        # Assign particle state
        particle.assign_state(new_pos=xs[-1], new_index=js[-1])

        # Return
        return particle, np.array(xs), np.array(js), np.array(log_likelihoods), total_visits, visits_x_adj, exceeds