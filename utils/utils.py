import numpy as np
import sys

def scale_likelihood(energy, beta, n, X):
    '''
    Function to scale energy

    @param observable : observable value
    @param beta : beta value
    @param n : number of live points
    @return element of PF sum
    '''

    # Set scale
    t = np.random.beta(n, 1, size=2)
    x_minus = X / t[0]
    x_plus = X * t[1]
    w_n = 0.5 * (x_minus - x_plus)

    # Return element of PF
    element = w_n * (1 + (np.finfo(np.float32).eps * (np.random.uniform() - 0.5))) * np.exp(- beta * energy)
    return element, x_minus

def compute_Z(en_list, beta, K):
    '''
    Function to compute partition function

    @param en_list : list of energies
    @param beta : value of inverse T
    @param K : number of live points
    @param nproc : number of processors
    @return Z : partition function
    '''

    # Iterate over energies
    Z = 0
    X = 1
    for n in range(len(en_list)):

        # Add to Z
        scale, X = scale_likelihood(en_list[n], beta, K, X)
        Z += scale

    # Return estimate of normalizing constant
    return Z

def compute_expectation(observable, beta, n, X):
    '''
    Function to compute energy fluctuation terms

    @param observable : observable value
    @param beta : beta value
    @param n : element index
    @param K : number of live points
    @return expectation value
    '''

    # Compute terms
    scale = scale_likelihood(observable, beta, n, X)
    exp = scale * observable

    # Return
    return exp

def compute_variance(observable, beta, n, X):
    '''
    Function to compute energy fluctuation terms

    @param observable : observable value
    @param beta : beta value
    @param n : element index
    @param K : number of live points
    @return term1, term2
    '''

    # Compute terms
    scale = scale_likelihood(observable, beta, n, X)
    var = scale * (observable ** 2)

    # Return
    return var

def compute_Cv(en_list, obs_list, beta, K):
    '''
    Function to compute constant volume heat capacity

    @param en_list : list of energies for weighing states
    @param obs_list : list of observables
    @param beta : beta factor (k_b*T)
    @param K : number of live points
    @return Cv : Heat capacity 
    '''

    # Compute factors
    Z = compute_Z(en_list, beta, K)
    mean = 0
    var = 0
    for n in range(len(obs_list)):

        mean += compute_expectation(obs_list[n], beta, n, K)
        var += compute_variance(obs_list[n], beta, n, K) 

    # Compute Cv
    mean /= Z
    var /= Z
    Cv = (beta ** 2) * (var - (mean**2))

    # Return 
    return Cv
