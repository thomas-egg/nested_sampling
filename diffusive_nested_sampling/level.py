import numpy as np

class Level(object):
    '''
    Level in DNS. Over the course of a DNS run we will create
    some fixed number of levels and the particle will traverse
    and sample them.
    '''
    def __init__(self, index, likelihood_boundary, prev_X = None):
        '''
        Initialize a DNS level

        @param index: index of level
        @param likelihood_boundary the minimum likelihood of the level
        @param init_X : Initial phase space volume estimate
        '''
        self.index = index 
        self.bound = likelihood_boundary
        self.visits = 0
        if prev_X is not None:
            self.X = prev_X * np.exp(-1)
        else:
            self.X = 1

    @property
    def likelihood_bound(self):
        '''
        Return lower limit of likelihood for level
        '''
        return self.bound

    def level_weight(self, j:float, l:float, max_level:int, chain_length:int):
        '''
        Exponentially decaying weight for this level

        @param j : current max level
        @param l : Lambda value for controlling backtracking
        '''
        if j < max_level - 1:
            weight = np.exp((self.index - j) / l)
        else:
            weight = 1.0
        weight /= self.get_X
        return weight, self.visits, weight * chain_length

    @property
    def get_X(self):
        '''
        Return phase space volume element
        '''
        return self.X

    def set_X(self, preceeding_X, chain, counter, C=1000):
        '''
        Compute phase space volume element of level given the history of 
        particle

        @param preceeding_X : phase space volume of preceeding level
        @param history : level/likelihood history of particles
        @param C : confidence
        '''
        # js = np.array(chain['j'])
        # ls = np.array(chain['L'])
        # inds = js == self.index - 1
        # numerator = np.sum(ls[inds] > self.bound) + (C * np.exp(-1))
        # denominator = counter[self.index - 1].item() + C
        # self.X = preceeding_X * (numerator / denominator)

    def set_visits(self, n_visits):
        self.visits = n_visits