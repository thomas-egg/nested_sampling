# Import libraries
import numpy as np
import unittest
from nested_sampling import NestedSampling, MCWalker, Replica
from pele.potentials import Harmonic
from mcpele.monte_carlo import SampleGaussian

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
        E = 0.5 * self.k * (np.dot(x, x))

        # Return
        return E

class TestParallel(unittest.TestCase):

    def setUp(self):

        # Initialize variables 
        self.pot = HarmonicPotential(2)
        self.r = [Replica(x, self.pot.get_energy(x)) for x in [np.random.uniform(low=-1, high=1, size=2) for _ in range(4)]]
        self.mc = MCWalker(self.pot)
        self.ns = NestedSampling(self.r, self.mc, iprint=1, nproc=2)

    def test_ns(self):

        # Test
        self.ns.run_sampler(1)
        for rep, new_rep in zip(self.r, self.ns.get_positions()):
            
            # Assert not equal
            self.assertNotEqual(rep.x[0], new_rep[0])