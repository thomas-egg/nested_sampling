import numpy as np
from scipy import constants
import csv
import sys

def compute_Cv(energies, beta, K):
    '''
    Function to compute constant volume heat capacity

    @param beta : beta factor (k_b*T)
    @param energies : energy list
    @param K : number of live points
    @param N : number of particles
    '''

    # Compute factors
    term1 = 0
    term2 = 0
    Z = 0
    scale = K/(K+1)

    # Iterate over 
    for n in range(len(energies)):

        # Add to Cv
        w_n = (scale ** (n+1)) - (scale ** (n+2))
        Z += w_n * np.exp(- beta * energies[n])
        term1 += w_n * np.exp(- beta * energies[n]) * (energies[n] ** 2)
        term2 += w_n * np.exp(- beta * energies[n]) * energies[n]

    # Post summation
    Cv = (beta ** 2) * ((term1/Z) - ((term2/Z)**2))

    # Return 
    return Cv