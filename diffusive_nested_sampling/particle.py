import torch
from level import Level
from typing import Callable

class Particle(object):
    '''
    Particle for DNS run. Each particle will explore
    the different likelihood levels of the system
    '''
    def __init__(self, init_pos:torch.tensor, init_index:torch.tensor):
        '''
        Initialize particle

        @param init_pos : initial position
        @param init_likelihood : initial likelihood
        '''
        
        self.x = init_pos
        self.j = init_index
        self.history = {
            'x' : self.x,
            'j' : self.j
        }                     

    @property
    def j(self):
        '''
        Return level index
        '''
        return self.j

    @property
    def pos(self):
        '''
        Get patrticle position
        '''
        return self.x
    
    def assign_state(self, new_pos:float, new_index:int):
        '''
        Change index
        
        @param new_index : index to jump to
        '''
        self.x = new_pos
        self.j = new_index