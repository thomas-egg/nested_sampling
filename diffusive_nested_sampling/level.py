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
        self.exp_visits = 0
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

    def level_weight(self, J:float, l:float, max_level:int):
        '''
        Exponentially decaying weight for this level

        @param j : current max level
        @param l : Lambda value for controlling backtracking
        '''
        if J < max_level:
            log_weight = (self.index - J) / l
        else:
            log_weight = np.log(1.0 / (max_level + 1))
        return log_weight

    @property
    def get_log_X(self):
        '''
        Return phase space volume element
        '''
        return self.log_X

    def set_log_X(self, prev_log_X: float, prev_j: int, prev_exceeds: int, C: int=1000):
        '''
        Compute phase space volume element of level given the history of 
        particle

        @param preceeding_X : phase space volume of preceeding level
        @param history : level/log_likelihood history of particles
        @param C : confidence
        '''
        numerator = prev_exceeds + (C * np.exp(-1))
        denominator = prev_j + C
        self.log_X = prev_log_X + np.log(numerator / denominator)

    def set_visits(self, total, x_adj, exceeds, exp_visits):
        self.total_visits += total
        self.visits_x_adj += x_adj
        self.exceeds += exceeds
        self.exp_visits += exp_visits

    def get_visits(self):
        return self.total_visits, self.exp_visits