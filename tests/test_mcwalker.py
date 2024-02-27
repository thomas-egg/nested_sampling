# Import MC
from nested_sampling import MCWalker, MCWalker_mcpele, random_displacement, Replica
import unittest
import numpy as np

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

# Class for testing Python MCMC
class TestMC(unittest.TestCase):

    def setUp(self):

        # Initialize variables 
        self.pot = HarmonicPotential(2)
        self.x = np.array([1.0, 1.0])
        self.r = Replica(self.x, self.pot.get_energy(self.x))
        self.Emax = 6
        self.mc = MCWalker(self.pot)

    def test_mc(self):

        # Test
        res = self.mc(self.r.x, 0.1, self.Emax, self.r.energy)
        print(res.x)
        self.assertNotEqual(res.x[0], self.x[0])

# Class for testing mcpele implementation
class TestMC_mcpele(unittest.TestCase):

    def setUp(self):

        # Initialize variables
        self.pot = HarmonicPotential(2)
        self.x = np.array([1.0, 1.0])
        self.r = Replica(self.x, self.pot.get_energy(self.x))
        self.Emax = 6
        self.nsamples = 100
        self.mc = MCWalker_mcpele(self.pot, x, 1, self.nsamples)