# Import libraries
import argparse
import matplotlib.pyplot as plt

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

    # Initialize variables
    boxlen = 2
    ndim = 2
    npoints = 100

    # Potential, replica list, MC walker, and finally NS
    pot = HarmonicPotential(ndim)
    replicas = [Replica(x, pot.get_energy(x)) for np.random.random(2) * boxlen in range(npoints)]
    mc = MCWalker(pot)
    ns = NestedSampling(replicas, mc)

    # Run sampler
    ns.run_sampler(100000)

    # Plot
    plt.hist()