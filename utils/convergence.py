import numpy as np
import sys
from .utils import *

def convergence_test(en_list, eps_list, K, nproc, iter_num, eps=10**-3):
    '''
    Function to test for algorithm convergence

    @param en_list : list of energies
    @param eps_list : tiebreaker
    @param K : number of live points
    @param nproc : number of processors
    @param iter_num : iteration number
    @eps : tolerance (eps<<1)
    @return : True or False depending on convergence
    '''

    # Get Z
    Z = compute_Z(en_list, 1, K, nproc)
    Z_live = ((K/(K+1)) ** iter_num) * max(en_list)

    # Return convergence
    return Z_live/Z < eps