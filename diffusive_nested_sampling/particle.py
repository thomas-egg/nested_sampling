from diffusive_nested_sampling.level import Level
from typing import Callable

class Particle(object):
    '''
    Particle for DNS run. Each particle will explore
    the different likelihood levels of the system
    '''
    def __init__(self, init_pos, init_index):
        '''
        Initialize particle

        @param init_pos : initial position
        @param init_likelihood : initial likelihood
        '''
        
        self.x = init_pos
        self.ind = init_index        

    @property
    def j(self):
        '''
        Return level index
        '''
        return self.ind

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
        self.ind = new_index