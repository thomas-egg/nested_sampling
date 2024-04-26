# Import libraries
import numpy as np
from nested_sampling import Replica, NestedSampling, MCWalker, spin_flip

class Ising2D(object):

    '''
    Pure Python Ising implementation for square 2D Ising model
    
    Parameters
    ----------
    J : float
        Coupling parameter for neighboring spins
    m0 : float
        Magnetic moment
    B : float
        External field
    ndim : int
        Number of dimensions
    '''

    def __init__(self, J=1, m0=0, B=0):

        # Initialize variables
        self.J = J
        self.m0 = m0
        self.B = B

    def getEnergy(self, x):
        '''
        Function to compute and return energy

        @param x : lattice site states
        @return E : energy
        '''

        # Compute energy
        E = 0
        N = len(x)
        for i in range(N):
            for j in range(N):

                # Add interaction term to E *PBC*
                E -= self.J * ((x[i][j] * x[i][(j+1) % N]) + (x[i][j] * x[(i+1) % N][j])) + (self.B * self.m0 * x[i][j])

        # Return
        return E

########
# MAIN #:q
########
if __name__ == '__main__':

    # Instantiate potential and initial conditions
    nlive = 1000
    pot = Ising2D()
    replicas = [Replica(x, pot.getEnergy(x), np.random.uniform(low=0.0, high=1.0)) for x in [np.random.choice([-1, 1], [12, 12]) for _ in range(nlive)]] 
    sampler = NestedSampling(replicas, pot, temp=1, stepsize=[-1,1], nproc=1, cpfreq=1000, verbose=False, enfile='ising_en12.txt', cpfile='ising_chk12.txt', use_mcpele=False, niter=30, takestep=spin_flip)
    sampler.run_sampler(500000)
