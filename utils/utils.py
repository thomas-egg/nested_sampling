import numpy as np
from joblib import Parallel, delayed
import sys

def scale_energy(energy, eps, beta, n, X):
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
    t = np.random.beta(n, 1, size=2)
    x_minus = X / t[0]
    x_plus = X * t[1]
    w_n = 0.5 * (x_minus - x_plus)

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

def compute_expectation(energy, eps, beta, n, K):
    '''
    Function to compute energy fluctuation terms

    @param energy : energy value
    @param eps : random value
    @param beta : beta value
    @param n : element index
    @param K : number of live points
    @return term1, term2
    '''

    # Set scale
    scale = K/(K+1)
    w_n = (scale ** (n+1)) - (scale ** (n+2))

    # Compute terms
    exp = w_n * (1 + (np.finfo(np.float32).eps * (eps - 0.5))) * np.exp(- beta * energy) * energy

    # Return
    return exp

def compute_variance(energy, eps, beta, n, K):
    '''
    Function to compute energy fluctuation terms

    @param energy : energy value
    @param eps : random value
    @param beta : beta value
    @param n : element index
    @param K : number of live points
    @return term1, term2
    '''

    # Set scale
    scale = K/(K+1)
    w_n = (scale ** (n+1)) - (scale ** (n+2))

    # Compute terms
    var = w_n * (1 + (np.finfo(np.float32).eps * (eps - 0.5))) * np.exp(- beta * energy) * (energy ** 2)

    # Return
    return var

def compute_Cv(en_list, eps_list, beta, K, nproc):
    '''
    Function to compute constant volume heat capacity

    @param beta : beta factor (k_b*T)
    @param energies : energy list
    @param eps_list : list of random values
    @param K : number of live points
    @param N : number of particles
    '''

    # Compute factors
    Z = compute_Z(en_list, eps_list, beta, K, nproc)

    # Serial 
    if nproc == 1:

        mean = 0
        var = 0
        for n in range(len(en_list)):

            mean += compute_expectation(en_list[n], eps_list[n], beta, n, K)
            var += compute_variance(en_list[n], eps_list[n], beta, n, K) 

    # Parallel
    else: 

        mean = np.sum(Parallel(n_jobs=nproc)(delayed(compute_expectation)(en_list[n], eps_list[n], beta, n, K) for n in range(len(en_list))))
        var = np.sum(Parallel(n_jobs=nproc)(delayed(compute_variance)(en_list[n], eps_list[n], beta, n, K) for n in range(len(en_list)))) 

    # Compute Cv
    mean /= Z
    var /= Z
    Cv = (beta ** 2) * (var - (mean**2))

    # Return 
    return Cv
