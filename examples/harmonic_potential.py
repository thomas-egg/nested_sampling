# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Import necessary functionality
from nested_sampling import NestedSampling, MCWalker, Replica

class HarmonicPotential(object):
    '''
    Harmonic potential energy function

    Parameters
    ----------
    k : float
        spring constant
    ndim : int
        number of dimensions
    '''

    def __init__(self, ndim, k=1):

        # Initialize class variables
        self.ndim = ndim
        self.k = k

    def get_energy(self, x):
        '''
        Function to compute and return energy

        @param x : position
        @return E : energy
        '''

        # Assert matching dimensions
        assert len(x) == self.ndim

        # Compute energy
        E = 0.5 * self.k * (x.dot(x))

        # Return
        return E

########
# MAIN #
########
if __name__ == '__main__':

    xer = np.array([1, 1.8])

    # Initialize variables
    boxlen = 3.0
    ndim = 2
    npoints = 100

    # Potential, replica list, MC walker, and finally NS
    pot = HarmonicPotential(ndim)
    print(pot.get_energy(xer))
    '''
    replicas = [Replica(x, pot.get_energy(x)) for x in [np.random.uniform(low=-boxlen, high=boxlen, size=ndim) for _ in range(npoints)]]
    mc = MCWalker(pot)
    ns = NestedSampling(replicas, mc)

    # Run sampler
    pos = np.array(ns.run_sampler(1000))

    # Plot
    x = pos[:,0]
    y = pos[:,1]

    plt.scatter(x[-100:], y[-100:])
    plt.savefig(f'bsplot.png')
    plt.scatter(x[:100], y[:100])
    plt.savefig(f'otherbsplot.png')
    '''