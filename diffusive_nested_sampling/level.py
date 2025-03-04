import numpy as np

class Level(object):
    '''
    Level in DNS. Over the course of a DNS run we will create
    some fixed number of levels and the particle will traverse
    and sample them.
    '''
    def __init__(self, index, log_likelihood_boundary, prev = None):
        '''
        Initialize a DNS level

        @param index: index of level
        @param log_likelihood_boundary the minimum log_likelihood of the level
        @param init_X : Initial phase space volume estimate
        '''
        self.index = index 
        self.bound = log_likelihood_boundary
        self.total_visits = 0
        self.visits_x_adj = 0
        self.exceeds = 0
        if prev is not None:
            self.log_X = prev - 1
        else:
            self.log_X = 0.0

    @property
    def log_likelihood_bound(self):
        '''
        Return lower limit of log_likelihood for level
        '''
        return self.bound

    def level_weight(self, j:float, l:float, max_level:int):
        '''
        Exponentially decaying weight for this level

        @param j : current max level
        @param l : Lambda value for controlling backtracking
        '''
        if j < max_level:
            log_weight = (self.index - j) / l
        else:
            log_weight = 0.0
        log_weight -= self.log_X
        return log_weight

    @property
    def get_log_X(self):
        '''
        Return phase space volume element
        '''
        return self.log_X

    def set_log_X(self, preceeding_log_X, C=1000):
        '''
        Compute phase space volume element of level given the history of 
        particle

        @param preceeding_X : phase space volume of preceeding level
        @param history : level/log_likelihood history of particles
        @param C : confidence
        '''
        numerator = self.exceeds + (C * np.exp(-1))
        denominator = self.visits_x_adj + C
        self.log_X = preceeding_log_X + np.log(numerator / denominator)

    def set_visits(self, total, x_adj, exceeds):
        self.total_visits += total
        self.visits_x_adj += x_adj
        self.exceeds += exceeds