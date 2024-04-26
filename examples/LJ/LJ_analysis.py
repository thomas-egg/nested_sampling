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
    with open(filename, newline='') as csvfile:           
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
    f1 = 'serial/en5_mcpele.txt'
    f2 = 'serial/en8_mcpele.txt'
    f3 = 'serial/en13_mcpele.txt'
    energies1 = get_energies(f1)
    energies2 = get_energies(f2)
    energies3 = get_energies(f3)

    # Temperatures to sample
    temps = np.linspace(0.05, 0.7, 1000)

    # Calculate Cv
    cv1 = [compute_Cv(energies1, 1/t, 300) for t in temps]
    cv2 = [compute_Cv(energies2, 1/t, 300) for t in temps]
    cv3 = [compute_Cv(energies3, 1/t, 300) for t in temps]

    # Plot
    plt.plot(temps, cv1, label='LJ5')
    plt.plot(temps, cv2, label='LJ8')
    plt.plot(temps, cv3, label='LJ13')
    plt.cm.viridis()
    plt.legend()
    plt.title('Heat Capacity (Constant Volume)')
    plt.xlabel(r'$\beta^{-1}$')
    plt.ylabel(r'$C_V$')
    plt.savefig('serial/CV.png')
