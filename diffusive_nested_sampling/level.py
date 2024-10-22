import torch
import numpy as np

class Level(object):
    '''
    Level in DNS. Over the course of a DNS run we will create
    some fixed number of levels and the particle will traverse
    and sample them.
    '''
    def __init__(self, index, likelihood_boundary):
        '''
        Initialize a DNS level

        @param index: index of level
        @param likelihood_boundary the minimum likelihood of the level
        @param init_X : Initial phase space volume estimate
        '''
        self.index = index 
        self.bound = likelihood_boundary
        self.X = np.exp(-self.index)

    @property
    def likelihood_bound(self):
        '''
        Return lower limit of likelihood for level
        '''
        return self.bound

    def level_weight(self, j:float, l:float):
        '''
        Exponentially decaying weight for this level

        @param j : current max level
        @param l : Lambda value for controlling backtracking
        '''
        weight = np.exp((self.index - j) / l)
        return weight

    @property
    def get_X(self):
        '''
        Return phase space volume element
        '''
        return self.X

    def set_X(self, preceeding_X, l_history, p_history, C=1000):
        '''
        Compute phase space volume element of level given the history of 
        particle

        @param preceeding_X : phase space volume of preceeding level
        @param history : level/likelihood history of particles
        @param C : confidence
        '''
        inds = p_history['j'] == self.index - 1
        numerator = torch.sum(l_history[inds] > self.bound) + (C * np.exp(-1))
        denominator = (torch.tensor(p_history['j']) == self.index).sum().item() + C
        self.X = preceeding_X * (numerator / denominator)