# Import libraries
import multiprocessing as mp
import numpy as np
from nested_sampling import Result

# For MC with mcpele
#from mcpele.monte_carlo import Metropolis_MCrunner
#from mcpele.monte_carlo import SampleGaussian

def random_displacement(x, stepsize):
    '''
    Function for taking random step

    @param x : position
    @param stepsize : size of step to take
    @return x : updated position
    '''

    # Take step
    x += np.random.uniform(low=-stepsize, high=stepsize, size=x.shape)

    # Return
    return x

def spin_flip(x, stepsize=None):
    '''
    Function for simple spin flip of random site 

    @param x : configuration
    '''

    # Flip spin
    i, j = np.random.randint(0, len(x), size=2)
    x[i][j] = - x[i][j]

    # Return
    return x

class MCWalker(object):
    '''
    Class for running a MCMC Sampler

    Parameters
    -----------
    potential :
        attribute of system with member function get_energy (in essence a
        particular potential energy function)
    x : array
        are the coordinates
    takestep : callable takestep object
        take a random montecarlo step, imported from pele: takestep(x) makes
        a move from x
    accept_test : list of callables
        it's an array of pointers to functions. The dereferenced functions
        operate a set of tests on the energy/configuration.
    events : list of callables
        it's an array of pointers to functions. This is general and not
        compulsury e.g. can use if you want to do something with the new
        configuration for the guy.

    Attributes
    ----------
    nsteps : integer
        tot. number of steps
    naccept : integer
        tot. number of accepted configurations
    xnew : array
        new proposed configuration
    accept : boolean
        true or false if energy constraint is satisfied
    '''

    def __init__(self, potential, takestep=random_displacement, accept_test=None, events=None, mciter=100):
        
        # Set variables
        self.potential = potential
        self.takestep = takestep
        self.accept_test = accept_test
        self.events = events
        self.mciter = mciter

    def __call__(self, x0, stepsize, Emax, energy, seed=None):

        # Run MC and return
        return self.run(x0, stepsize, Emax, energy, seed=seed)

    def run(self, x0, stepsize, Emax, energy, seed=None):
        '''
        Function to run Monte Carlo algorithm

        @param x0 : initial positions
        @param stepsize : size of step to take
        @param Emax : max energy for acceptance
        @param energy : energy of configuration
        @param seed : random seed
        '''

        # Test if energy is reasonable
        assert energy <= Emax

        # Variables for loop
        x = x0.copy()
        naccept = 0

        # If accept test is given run it
        if self.accept_test is not None:
            if not self.accept_test(x, 0.):
                raise Exception("initial configuration for monte carlo chain failed configuration test")
        
        # Run for predetermined number of steps
        for i in range(self.mciter):

            # New x
            xnew = x.copy()

            # Take step
            xnew = self.takestep(xnew, stepsize)

            # Get energy
            e = self.potential.getEnergy(xnew)

            # Acceptance criteria/configuration tests
            accept = e < Emax
            if accept and self.accept_test is not None:
                accept = self.accept_test(xnew, e)

            # Conditional based on acceptance
            if accept:

                # Update state
                x = xnew
                energy = e
                naccept += 1

            # process callback functions if they exist
            if self.events is not None:
                for event in self.events:
                    event(coords=x, x=x, energy=energy, accept=accept)

        # Get results
        res = Result()
        res.Emax = Emax
        res.energy = energy
        res.x = x
        res.nsteps = self.mciter
        res.naccept = naccept

        # Return
        return res

#class MCWalker_mcpele(Metropolis_MCrunner):
#    '''
#    MCWalker with mcpele
#
#    Parameters
#    ----------
#    potential : pele potential
#        potential energy function to evaluate at steps
#    x : array
#        coordinates of the walker (supply initial, these get updated)
#    temperature : float
#        temperature to conduct sampling at, if athermal use 1
#    nsteps : int
#        number of steps to take
#    '''
#
#    def set_control(self, temp):
#        '''
#        Function to set temperature of system
#
#        @param temp : temperature to set
#        '''
#
#        # Set temp
#        self.temperature = temp
