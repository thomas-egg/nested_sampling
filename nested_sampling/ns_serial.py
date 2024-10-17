'''
Implementation for serial nestes sampling
'''

# Import libraries
import random
import copy
import numpy as np
import sys
import csv
import multiprocessing as mp
from nested_sampling import MCWalker, MCWalker_mcpele, Result, random_displacement
from utils import convergence_test
import hickle as hkl

def _mc_runner(r, potential, temperature, stepsize, niter, Emax, use_mcpele, takestep):
    '''
    Function to run MC with/without mcpele
    @param r : replica
    @param potential : potential energy function
    @param temperature : temperature
    @param stepsize : size of step
    @param niter : number steps
    @param Emax : maximum energy for MC
    @param use_mcpele : bool for using mcpele or not
    @param takestep : type of step to take
    @return res : result
    '''

    # mcpele
    if use_mcpele:

        res = Result()
        mc = MCWalker_mcpele(potential, np.array(r.x), temperature, stepsize, niter, hEmax=Emax, hEmin=-100, radius=0)
        mc.run()
        res.x = mc.get_coords()
        res.energy = mc.get_energy()
        res.Emax = Emax
        res.naccept = mc.get_accepted_fraction() * niter
        res.nsteps = niter

    # python implementation otherwise
    else:

        mc = MCWalker(potential, takestep=takestep, mciter=niter)
        res = mc(r.x, stepsize, Emax, r.energy)

    # Return result
    return res

class Replica(object):
    '''
    Class for representing a live point

    Parameters
    ----------
    x : array
        the coordinates
    energy : float
        configurational energy of the point
    niter : int
        number of MC iterations for this point
    from_random : bool
        boolean for if this walker was spawned randomly
    '''

    def __init__(self, x, energy, eps=0, niter=0, from_random=True):

        # Copy values
        self.x = x
        self.energy = energy
        self.niter = niter
        self.from_random = from_random

    def copy(self):
        '''
        Return a copy of the replica
        
        @return copy
        '''

        # Return
        return copy.deepcopy(self)

class NestedSampling(object):
    '''
    Class for implementing the Nested Sampling algorithm

    Parameters
    ----------
    replicas : list
        list of objects of type Replica
    mc_walker: callable
        class of type MonteCarloWalker or similar. It
        performs a Monte Carlo walk to sample phase space.
        It should return an object with attributes x, energy, nsteps, naccept, etc.
    verbose : bool
        print status messages
    iprint : int
        if verbose is true, then status messages will be printed every iprint iterations
    cpfile: str
        checkpoint file name
    cpfreq: int
        checkpointing frequency in number of steps
    cpstart: bool
        start calculation from checkpoint binary file
    dispatcher_URI: str
        address (URI) of dispatcher (required for distributed parallelisation)

    Attributes
    ----------
    nreplicas : integer
        number of replicas
    stepsize : float
        starting stepsize. It is then adapted to meet some target MC acceptance
        by default 0.5
    max_energies : list
        array of stored energies (at each step the highest energy configuration
        is stored and replaced by a valid configuration)
    store_all_energies: bool
        store the energy of all the nproc replicas replaced when running NS
        in parallel
    '''

    def __init__(self, replicas, pot, temp, stepsize=0.1, nproc=1, verbose=False, max_stepsize=1.0, iprint=1000, chkpt=True, cpfile=None, cpfreq=10000, enfile=None, use_mcpele=False, niter=100, takestep=random_displacement):

        # Initialize class variables
        self.verbose = verbose
        self.iprint = iprint
        self.replicas = replicas
        self.nreplicas = len(self.replicas)
        self.stepsize = stepsize      
        self.max_stepsize = max_stepsize
        self.takestep = takestep
        self.chkpt = chkpt
        self.cpfreq = cpfreq
        self.cpfile = cpfile
        self.enfile = enfile
        self.max_energies = []
        self.store_all_energies = True        
        self.iter_number = 0
        self._mc_niter = niter # total number of monte carlo iterations
        self.pot = pot
        self.temperature = temp

        # use mcpele or not
        self.use_mcpele = use_mcpele
        if self.use_mcpele:

            # Set up sampler for MC
            self.backend = 'threads'

        else:
            self.backend = 'loky'
        
        
    ####################
    # Functions for MC #
    ####################

    def _MC(self, r, Emax):
        '''
        Function to run serial Monte Carlo WITHOUT parallelization

        @param r : coordinates of replica
        @param Emax : max energy threshold
        @return res : result
        '''
        
        # Position
        r = r[0]

        # Run MC
        res = _mc_runner(r, self.pot, self.temperature, self.stepsize, self._mc_niter, Emax, self.takestep)

        # Return
        return [res]

    def _run_MC(self, Emax, r):
        '''
        Function to run MC.

        @param r : replica to propagate
        @param Emax : maximum energy
        @return r : updated walker
        @return res : MC result
        ''' 

        # Save result
        rsave = r
        res = self._MC(r, Emax)

        # Update
        for replica, result in zip(r, res):

            # Save
            replica.x = result.x
            replica.energy = result.energy
            replica.niter = result.nsteps
        
        # If verbose, print data
        if self.verbose:

            # Print statistics
            print(f'step: {self.iter_number}')
            print(f'%accept: {float(res.naccept / res.nsteps)}')
            print(f'Enew: {res.energy}')
            print(f'Eold: {rsave.energy}')
            print(f'stepsize: {self.stepsize}')

        if isinstance(self.stepsize, float):
            self._adjust_stepsize(res)

        # Return
        return r, res

    ###########################
    # Nested Sampling Updates #
    ###########################

    def _pop_replica(self):
        '''
        Function to remove replica with highest energy (lowest likeliehood)
        '''

        # Pop!
        self.replicas.pop()

    def _add_replica(self, rlist):
        ''' 
        Function to add replica

        @param rlist : replica to add
        '''

        # Append
        for r in rlist:
            
            # Add replica
            self.replicas.append(r)

        # Sorts
        self._sort_replicas()

    def _get_starting_configurations_from_replicas(self):
        '''
        Use existing replicas as starting configurations
        '''
        # choose a replica randomly
        assert len(self.replicas) == self.nreplicas

        # If serial
        rlist = random.sample(self.replicas, 1)
        self.starting_replicas = rlist
        
        # make a copy of the replicas so we don't modify the old ones
        rlist = [r.copy() for r in rlist]
        return rlist

    def _adjust_stepsize(self, res):
        '''
        If acceptance ratio is drifting low we can adjust the stepsize
        so that we continue efficient sampling

        @param res : result to base adjustment off of
        '''

        # If we have no stepsize exit
        if self.stepsize is None:
            return

        # Set factor to adjust by and ratios
        f = 0.8
        target_ratio = 0.5
        current_ratio = float(sum(r.naccept for r in res)) / sum(r.nsteps for r in res)

        # Test if we are near target
        if current_ratio < target_ratio:

            # Adjust stepsize
            self.stepsize *= f

        else:

            # Adjust stepsize
            self.stepsize /= f

        # Test max stepsize
        if self.stepsize > self.max_stepsize:

            # Set to max
            self.stepsize = self.max_stepsize

    def _sort_replicas(self):
        '''
        Sort replicas in decreasing energy order
        '''

        # Sort
        self.replicas.sort(key=lambda r : (r.energy))

        # If we store energies
        if self.store_all_energies:
            self.max_energies.append(self.replicas[-1].energy)

    def _get_new_Emax(self):
        '''
        Function to return new Emax

        @return new Emax
        '''
        self._sort_replicas()
        return self.replicas[-1].energy


    def _one_iteration(self):
        '''
        Function to run one iteration of the NS algorithm
        '''
        # New Emax
        Emax = self._get_new_Emax()

        # Pop
        r = self._get_starting_configurations_from_replicas()
        self.__pop_replica()

        # Run MC for replica
        rnew, res = self._run_MC(Emax, r)

        # Finish off step
        self._add_replica(rnew)

        # Test to make sure sizes match
        if self.nreplicas != len(self.replicas):

            # Throw error
            raise Exception('Size mismatch in number of replicas!')

    #######
    # RUN #
    #######

    def run_sampler(self, steps):
        '''
        Function to run NS until convergence criteria is met

        IDEA FOR NOW (Skilling et al. 2006):
        if Z_live / Z_i <= eps -> TERMINATE

        @param eps : epsilon for termination
        '''

        # Begin
        print('############')
        print('# NS BEGIN #')
        print('############')

        # While termination condition not met
        i = 0
        while i <= steps:

            # ITERATE
            self._one_iteration()
            self.iter_number += 1

            # Run
            if steps > 0:
                i += 1
            else:
                if self.iter_number % self.cpfreq == 0:

                    # Convergence test
                    criteria = convergence_test(self.max_energies[:][0], self.max_energies[:][1], self.nreplicas, 1, self.iter_number)

                    # Break if converged
                    if criteria:
                        print('# CONVERGENCE #') 
                        break

            print(f'# ITERATION {self.iter_number} #')

            # Write to checkpoint
            if self.chkpt and self.iter_number % self.cpfreq == 0:
                self.write_out(self.get_positions(), self.cpfile)
                self.write_out(self.max_energies, self.enfile)

        # Return/End
        print('##########')
        print('# NS END #')
        print('##########')

    ########
    # DATA #
    ########

    def get_positions(self):
        '''
        Function to return positions fo the replicas

        @return pos : replica positions
        '''

        # Iterate
        pos = []
        for r in self.replicas:

            # save
            pos.append(r.x.tolist())

        # Return
        return pos

    def write_out(self, data, fi):
        '''
        Function to write out data

        @param data : data to write out
        @param fi : file to write to
        '''
        hkl.dump(fi, data)
