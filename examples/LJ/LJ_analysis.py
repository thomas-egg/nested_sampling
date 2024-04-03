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

def get_energies(filename):
    '''
    Parse file for energies

    @param filename : name of saved energy file
    @return energies : energy data
    '''

    # Read in data
    energies = []
    with open('en_5_new.txt', newline='') as csvfile:           
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for element in row:
                energies.append(ast.literal_eval(element.strip()))

    # Return
    return energies

########
# MAIN #
########

if __name__ == '__main__':

    # Read data
    filename = 'en_5_new.txt'
    energies = get_energies(filename)

    # Temperatures to sample
    temps = np.linspace(0.02, 0.7, 1000)

    # Calculate Cv
    cv = [compute_Cv(energies, 1/t, 300) for t in temps]

    # Plot
    plt.plot(temps, cv)
    plt.title('Heat Capacity (Constant Volume)')
    plt.xlabel(r'$\beta^{-1}$')
    plt.ylabel(r'$C_V$')
    plt.savefig('CV.png')
