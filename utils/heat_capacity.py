from utils import compute_Z
from joblib import Parallel, delayed
import numpy as np
import csv
import sys

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
