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
    '''
    HOW THIS WORKS
    --------------

    This is an example of how one might use Nested Sampling (NS) to calculate the 
    evidence (partition function) associated with a simple 2-D Gaussian (harmonic potential).
    We define an exponentiated harmonic potential as the Likelihood, and enforce a 
    uniform spherical prior:

    prior = (D/2)!/(pi^(D/2))

    Analytical calculations with k = 1/(sigma^2):
    Z = 0.02
    log(Z) = -3.91

    We then run the NS algorithm for some steps and see how we did.
    '''

    # Initialize variables
    boxlen = 1.0
    ndim = 2
    npoints = 100

    # Potential, replica list, MC walker, and finally NS
    pot = HarmonicPotential(ndim, k=1/(0.1**2))
    
    replicas = [Replica(x, pot.get_energy(x)) for x in [np.random.uniform(low=-boxlen, high=boxlen, size=ndim) for _ in range(npoints)]]
    mc = MCWalker(pot)
    ns = NestedSampling(replicas, mc)

    # Run sampler
    Z, w, l = ns.run_sampler(1500)

    # Plot
    plt.ylabel('Likelihood')
    plt.xlabel('$Log(X)$')
    plt.plot(np.log(w), l, marker='o')
    plt.savefig('likelihood_vs_logX.png')
    