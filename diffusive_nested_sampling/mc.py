import numpy as np
from diffusive_nested_sampling.particle import Particle
from diffusive_nested_sampling.level import Level

'''
Test MCMC shamelessly stolen from original Diffusive
Nested Sampling Paper: https://arxiv.org/pdf/0912.2380. 
'''

class MCMC(object):
    def __init__(self, beta, log_likelihood_function, max_J, iterations=10000, C=1000):
        '''
        Simple Monte Carlo implementation

        @param beta : exponent
        @param C : confidence
        '''
        self.beta = beta
        self.log_likelihood_function = log_likelihood_function
        self.max_J = max_J
        self.iters = iterations
        self.C = C

    def __call__(self, particle:Particle, levels, J):
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
        for i in range(self.iters):

            # Set up proposal - Jeffreys Prior
            S = 10**np.random.uniform(np.log10(1.0), np.log(10))
            S_prime = 10**np.random.uniform(np.log10(1.0), np.log10(100.0))

            # Proposals
            ind = np.random.randint(x.shape, size=2)
            step = np.zeros_like(x)
            step[ind] = np.random.uniform(-1/S, 1/S, size=2)
            x_new = np.clip(x + step, -0.5, 0.5)
            j_new = int(np.clip(np.random.normal(loc=j, scale=S_prime), 0, J))

            # Compute likelihoods
            new_logL = self.log_likelihood_function(x_new)
            if new_logL > levels.get_level(j).log_likelihood_bound:
                x = x_new
            else:
                new_logL = self.log_likelihood_function(x)
                x = x

            # Update level
            a = (new_logL > levels.get_level(j_new).log_likelihood_bound) * levels.get_acceptance_ratio(j, j_new, self.beta)
            r = min(1, a)
            u = np.random.rand()
            if u < r:
                j = j_new
            else:
                j = j

            # Accumulate quantities
            if j < J:
                visits_x_adj[j] += 1
                if new_logL > levels.get_level(j+1).log_likelihood_bound:
                    exceeds[j] += 1
            total_visits[j] += 1
            log_likelihoods.append(new_logL)

            # Thin chain
            if i % 10000 == 0:
                xs.append(x)
                js.append(j)
        
        # Assign particle state
        particle.assign_state(new_pos=x, new_index=j)

        # Return
        return particle, np.array(xs), np.array(js), np.array(log_likelihoods), total_visits, visits_x_adj, exceeds