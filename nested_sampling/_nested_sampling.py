# Import libraries
import random
import numpy as np
import sys
import multiprocessing as mp

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

    def __init__(self, x, energy, niter=0, from_random=True):

        # Copy values
        self.x = x
        self.energy = energy
        self.niter = niter
        self.from_random = from_random

    def copy(self)
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
    nproc : int
        number of processors to use for parallel nested sampling
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
    serializer: str
        choice of serializer

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

    def __init__(self, replicas, mc_walker, stepsize=0.1, nproc=1, verbose=True, max_stepsize=0.5, iprint=1, cpfile=None, cpfreq=10000, cpstart = False, dispatcher_URI=None, serializer='pickle'):

        # Initialize class variables
        self.nproc = nproc
        self.verbose = verbose
        self.iprint = iprint
        self.replicas = replicas
        self.nreplicas = len(self.replicas)
        self.sort_replicas()
        self.mc_walker = mc_walker
        self.stepsize = stepsize
        self.max_stepsize = max_stepsize
        self.cpfreq = cpfreq
        self.cpfile = cpfile
        self.cpstart = cpstart
        self.max_energies = []
        self.store_all_energies = True        
        self.iter_number = 0
        self.failed_mc_walks = 0
        self._mc_niter = 0 # total number of monte carlo iterations

        # Evidence/Partition Function
        self.Z = 0.0
        
    ####################
    # Functions for MC #
    ####################

    def do_serial_MC(self, r, Emax):
        '''
        Function to run MC on a replica WITHOUT parallelization.

        @param r : replica to propagate
        @param Emax : maximum energy 
        @return r : updated walker
        @return res : MC result
        ''' 

        # Perform with one processor
        assert self.nproc == 1

        # Save initial replica and random seed
        rsave = r
        seed = np.randim.randint(0, sys.maxint)

        # Run walk
        res = self.mc_walker(r.x, self.stepsize, Emax, r.energy, seed)

        # Update replica
        r.x = res.x
        r.energy = res.energy
        r.niter += res.nsteps
        self.adjust_stepsize(res)

        # If verbose, print data
        if verbose:

            # Print statements
            print(f'step: {self.iter_number}')
            print(f'%accept: {float(res.naccept / res.nsteps)}')
            print(f'Enew: {res.energy}')
            print(f'Eold: {rsave.energy}')
            print(f'stepsize: {self.stepsize}')

        # Return
        return r, res

    def serial_MC_chain(self, configs, Emax):
        '''
        Function to run MC for all replicas 

        @param configs : configurations of all replicas
        @param Emax : the maximum energy
        '''

        # Set up run
        assert len(configs) == self.nproc

        # Run MC - no parallelization
        if self.nproc == 1:
            
            # Run MC
            rnew, res = do_serial_MC(configs[0], Emax)
            rnew_list = [rnew]
            res = [res]

        # Add steps
        self._mc_iter += sum((result.nsteps for result in res))

        # Loop over results and configurations
        for result, r in zip(res, configs):

            # Count failed walks
            if result.naccept == 0:

                # Increment
                self.failed_mc_walks += 1
                sys.stdout.write("WARNING: step: %d accept %g Enew %g Eold %g Emax %g Emin %g stepsize %g\n" % 
                                 (self.iter_number, float(result.naccept) / result.nsteps,
                                  result.energy, r.energy, Emax, self.replicas[0].energy,
                                  self.stepsize))
        
        # Adjust stepsize and return new positions
        self.adjust_stepsize(res)
        return rnew_list       

    ###########################
    # Nested Sampling Updates #
    ###########################

    def pop_replica(self):
        '''
        Function to remove replica with highest energy (lowest likeliehood)
        '''

        # Pop!
        r = self.replicas.pop()

        # If we store energies
        if self.store_all_energies:
            self.max_energies.append(r.energy)

    def add_replica(self, rlist):
        ''' 
        Function to add replica

        @param rlist : replica to add
        '''

        # Append
        for r in rlist:
            self.replicas.append(r)

        # Sort
        self.sort_replicas()

    def sort_replicas(self):
        '''
        Sort replicas in decreasing energy order
        '''

        # Sort
        self.replicas.sort(key = lambda r : r.energy)

    def get_starting_configurations_from_replicas(self):
        '''
        Use existing replicas as starting configurations
        '''
        # choose a replica randomly
        assert len(self.replicas) == (self.nreplicas - self.nproc)
        rlist = random.sample(self.replicas, self.nproc)
        self.starting_replicas = rlist
        
        # make a copy of the replicas so we don't modify the old ones
        rlist = [r.copy() for r in rlist]
        return rlist

    def adjust_stepsize(self, res):
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
        current_ratio = float(sum(r.naccept for r in res)) / sum(r.stsps for r in res)

        # Test if we are near target
        if current_ratio < target_ratio:

            # Adjust stepsize
            self.stepsize *= f

        else:

            # Adjust stepsize
            self.stepsize /= f

    def get_new_Emax(self)
        '''
        Function to return new Emax

        @return new Emax
        '''
        return self.replicas[-self.nproc].energy


    def one_iteration(self):
        '''
        Function to run one iteration of the NS algorithm
        '''

        # New Emax
        Emax = self.get_new_Emax()

        # Pop
        self.sort_replicas()
        self.pop_replicas()

        # Run MC for replica
        r = self.get_starting_configurations_from_replicas
        rnew = self.serial_MC_chain()

        # Finish off step
        self.add_replica(r)
        self.iter_number += 1

        # Add to Z
        self.Z += (np.exp(-Emax)) * (1 / (self.iter_number + 1))

        # Test to make sure sizes match
        if self.nreplicas != len(self.replicas):

            # Throw error
            raise Exception('Size mismatch in number of replicas!')

    #######
    # RUN #
    #######

    def run_sampler(steps):
        '''
        Function to run NS until convergence criteria is met

        IDEA FOR NOW (Skilling et al. 2006):
        if Z_live / Z_i <= eps -> TERMINATE

        @param eps : epsilon for termination
        '''

        # While termination condition not met
        i = 0
        while i < steps:

            # Run
            self.one_iteration()
