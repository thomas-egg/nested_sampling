'''
Function to plot LJ Nested Sampling outputs and observables
'''

# Import libraries
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import csv
import ast
import sys
from utils import compute_Cv
import argparse

def get_energies(filename):
    '''
    Parse file for energies

    @param filename : name of saved energy file
    @return energies : energy data
    '''

    # Read in data
    data = np.loadtxt(filename, dtype=float, delimiter=',')
    energies = data[:,0]
    eps = data[:,1]

    # Return
    return energies, eps

########
# MAIN #
########

if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser(description='Nested Sampling on Ising model in 2D')
    parser.add_argument('-K', metavar='--nlive', type=int, help='Number of live points')
    parser.add_argument('-N', metavar='--nproc', type=int, help='Number of processors')
    args = parser.parse_args()

    # Read data
    nproc = args.N
    nlive = args.K
    f1 = 'ising_en8.txt'
    f2 = 'ising_en12.txt'
    f3 = 'ising_short_en6.txt'
    #en1, eps1 = get_energies(f1)
    en2, eps2 = get_energies(f2)
    #en3, eps3 = get_energies(f3)

    # Temperatures to sample
    temps = np.linspace(1, 6, 100)

    # Calculate Cv
    #cv1 = [compute_Cv(en1, eps1, 1/t, nlive, nproc)/64 for t in temps]
    cv2 = [compute_Cv(en2, eps2, 1/t, 2000, nproc)/144 for t in temps]
    #cv3 = [compute_Cv(en3, 1/t, nlive, nproc)/36 for t in temps]
 
    # Plot
    #plt.plot(temps, cv1, label='8x8', color='lightseagreen')
    plt.plot(temps, cv2, label='12x12', color='teal')
    #plt.plot(temps, cv3, label='6x6', color='mediumaquamarine')
    plt.vlines(2.269, 0, 1.5, linestyle='dashed', color='black')
    plt.legend()
    plt.title('Heat Capacity')
    plt.xlabel(r'$\beta^{-1}$')
    plt.ylabel(r'$C_V$')
    plt.savefig('CV.png')
