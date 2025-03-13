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

    def particle_update(self, x, level_j):

        # Proposal
        s = np.random.randn() / np.sqrt(-np.log(np.random.rand()))
        ind = np.random.randint(x.shape, size=1)
        step = np.zeros_like(x)
        step[ind] = (10 ** (1.5 - 3 * np.abs(s))) * np.random.randn()
        x_new = np.clip(x + step, -0.5, 0.5)

        # Compute likelihoods
        new_logL = self.log_likelihood_function(x_new)
        if new_logL > level_j.log_likelihood_bound:
            x = x_new
        else:
            new_logL = self.log_likelihood_function(x)
            x = x
        return x, new_logL

    def level_update(self, j, levels, likelihood):

        # Proposals
        j_new = j + int(np.random.randn() * (10 ** (2 * np.random.rand())))
        j_new = (j_new + levels.current_max_J + 1) % (levels.current_max_J + 1) # Wrap around
        if j_new == j and 0 < j_new < levels.current_max_J:
            j_new += np.random.choice([-1, 1])

        # Update level
        a = (likelihood > levels.get_level(j_new).log_likelihood_bound) * levels.get_acceptance_ratio(j, j_new, self.beta)
        r = min(1, a)
        u = np.random.rand()
        if u <= r:
            j = j_new
        else:
            j = j
        return j

    def __call__(self, particle:Particle, levels):
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
        total_visits = np.zeros(levels.current_max_J + 1)
        x_adj_visits = np.zeros(levels.current_max_J + 1)
        exceeds = np.zeros(levels.current_max_J + 1)
        j, x = particle.j, particle.pos
        i = 0
        condition = lambda current_max_J: (len(log_likelihoods) < self.iters if current_max_J < self.max_J else i < self.iters)
        while condition(levels.current_max_J):

            # MCMC step
            order = np.random.rand()
            if order <= 0.5:
                x, new_logL = self.particle_update(x, levels.get_level(j))
                j = self.level_update(j, levels, new_logL)
            else:
                j = self.level_update(j, levels, self.log_likelihood_function(x))
                x, new_logL = self.particle_update(x, levels.get_level(j))

            # Accumulate quantities
            total_visits[j] += 1
            if j < levels.current_max_J:
                x_adj_visits[j] += 1
                if new_logL > levels.get_level(j + 1).log_likelihood_bound:
                    exceeds[j] += 1

            # Thin chain
            if new_logL > levels.get_level(levels.current_max_J).log_likelihood_bound:
                log_likelihoods.append(new_logL)
            if i % 1000 == 0:
                xs.append(x)
                js.append(j)
            
            # Increment i
            i += 1
        
        # Assign particle state
        particle.assign_state(new_pos=x, new_index=j)

        # Return
        return particle, np.array(xs), np.array(js), np.array(log_likelihoods), total_visits, x_adj_visits, exceeds