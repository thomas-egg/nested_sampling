import numpy as np
from joblib import Parallel, delayed
import sys

def scale_energy(energy, eps, beta, n, K):
    '''
    Function to scale energy

    @param energy : energy value
    @param eps : random value to break degeneracy
    @param beta : beta value
    @param n : element index
    @param K : number of live points
    @return element of PF sum
    '''

    # Set scale
    scale = K/(K+1)
    w_n = (scale ** (n+1)) - (scale ** (n+2))

    # Return element of PF
    element = w_n * (1 + (np.finfo(np.float32).eps * (eps - 0.5))) * np.exp(- beta * energy)
    return element

def compute_Z(en_list, eps_list, beta, K, nproc):
    '''
    Function to compute partition function

    @param en_list : list of energies
    @param beta : value of inverse T
    @param K : number of live points
    @param nproc : number of processors
    @return Z : partition function
    '''

    # Serial
    if nproc == 1:

        # Iterate over energies
        Z = 0
        for n in range(len(en_list)):

            # Add to Z
            Z += scale_energy(en_list[n], eps_list[n], beta, n, K)

    # Parallel
    else:

        # Compute Z
        Z = sum(Parallel(n_jobs=nproc)(delayed(scale_energy)(en_list[n], eps_list[n], beta, n, K) for n in range(len(en_list))))

    # Return
    return Z
