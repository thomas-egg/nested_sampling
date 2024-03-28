# Import libraries
import random
import copy
import numpy as np
import sys
import csv
import multiprocessing as mp
from joblib import Parallel, delayed

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

    def __init__(self, replicas, mc_walker, stepsize=0.1, nproc=1, verbose=False, max_stepsize=0.5, iprint=100, chkpt=False, cpfile=None, cpfreq=1000, enfile=None, cpstart = False, dispatcher_URI=None, serializer='pickle', use_mcpele=False, sampler=None):

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
        self.chkpt = chkpt
        self.cpfreq = cpfreq
        self.cpfile = cpfile
        self.cpstart = cpstart
        self.enfile = enfile
        self.max_energies = []
        self.store_all_energies = True        
        self.iter_number = 0
        self.failed_mc_walks = 0
        self._mc_niter = 0 # total number of monte carlo iterations
        self.xqueue = [1] # queue for computing the weight associated with a given likelihood contour
        self.Eold = 0

        # use mcpele or not
        self.use_mcpele = use_mcpele
        if self.use_mcpele:

            # Set up sampler for MC
            self.step = sampler
        
        
    ####################
    # Functions for MC #
    ####################

    def serial_MC(self, r, Emax):
        '''
        Function to run serial Monte Carlo WITHOUT parallelization

        @param r : coordinates of replica
        @param Emax : max energy threshold
        @return res : result
        '''
        
        # Position
        r = r[0]

        # Save initial replica and random seed
        rsave = r
        seed = np.random.randint(0, 1000)

        # mcpele implementation
        if self.use_mcpele:

            # Run MC
            self.mc.set_takestep(self.stepsize)
            res = self.mc.run()

        # Pure Python
        else:

            # Run walk
            res = self.mc_walker(r.x, self.stepsize, Emax, r.energy, seed)

        # Return
        return [res]

    def parallel_MC(self, Emax):
        '''
        Function to run parallelized MC

        @param r : coordinate list
        @param Emax : max energy threshold
        '''

        # Random seed
        seed = np.random.randint(0, 1000)

        # mcpele
        if self.use_mcpele:
            
            # Run MC
            self.mc.set_takestep(self.stepsize)
            res = Parallel(n_jobs=self.nproc, prefer="threads")(delayed(self.mc.run()) for r in self.replicas)

        else:

            # Run MC
            res = Parallel(n_jobs=self.nproc)(delayed(self.mc_walker)(r.x, self.stepsize, Emax, r.energy, seed) for r in self.replicas)
            
        # Return
        return res

    def run_MC(self, r, Emax):
        '''
        Function to run MC.

        @param r : replica to propagate
        @param Emax : maximum energy
        @return r : updated walker
        @return res : MC result
        ''' 

        # Perform with one processor
        if self.nproc == 1:

            # Save result
            res = self.serial_MC(r, Emax)

            # Update replica
            r = r[0]
            r.x = res.x
            r.energy = res.energy
            r.niter += res.nsteps


        # Perform with parallelization
        else:

            # Save results
            res = self.parallel_MC(Emax)

            # Update
            for replica, result in zip(r, res):

                # Save
                replica.x = result.x
                replica.energy = result.energy
                replica.niter = result.nsteps
            
        # Update stepsize
        self.adjust_stepsize(res)
        
        # If verbose, print data
        if self.verbose:

            # Print statistics
            print(f'step: {self.iter_number}')
            print(f'%accept: {float(res.naccept / res.nsteps)}')
            print(f'Enew: {res.energy}')
            print(f'Eold: {rsave.energy}')
            print(f'stepsize: {self.stepsize}')

        # Return
        return r, res

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
        assert len(self.replicas) == self.nreplicas

        # If serial
        if self.nproc == 1:
            rlist = random.sample(self.replicas, 1)

        # Parallel 
        else:
            rlist = self.replicas

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

    def get_new_Emax(self):
        '''
        Function to return new Emax

        @return new Emax
        '''
        return self.replicas[-1].energy


    def one_iteration(self):
        '''
        Function to run one iteration of the NS algorithm
        '''
        # New Emax
        Emax = self.get_new_Emax()
        r = self.get_starting_configurations_from_replicas()

        # Serial
        if self.nproc == 1:

            # Pop
            self.sort_replicas()
            self.pop_replica()

            # Run MC for replica
 
            rnew, res = self.run_MC(r, Emax)

            # Finish off step
            self.add_replica([rnew])
            
        # Parallel
        else:

            # Run mc
            self.replicas, res = self.run_MC(r, Emax)
            self.sort_replicas()

        # Update step
        self.iter_number += 1

        # Test to make sure sizes match
        if self.nreplicas != len(self.replicas):

            # Throw error
            raise Exception('Size mismatch in number of replicas!')

        # Save energy from this step
        self.Eold = Emax

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

        # Histogram
        pos = []

        # Begin
        print('############')
        print('# NS BEGIN #')
        print('############')

        # While termination condition not met
        i = 0
        while i < steps:

            # Run
            self.one_iteration()
            i += 1

            # Print out/save energy
            if i % self.iprint == 0:
                print(self.Eold)

            # Write to checkpoint
            if self.chkpt and i % self.cpfreq == 0:
                self.write_out([i, self.get_positions()], self.cpfile)
                self.write_out([i, self.Eold], self.enfile)

            #if self.verbose and i % self.iprint == 0:
            print(self.get_positions())

        # Return
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

    def compound_Z(self, Eold):

        # Add to Z (trapezoid rule)
        w = .5 * (self.xqueue[0] - self.xqueue[2])
        l = np.exp(-Eold)
        self.Z += l * w
        self.L.append(l)
        self.Zlist.append(l*w)
        self.w.append(self.xqueue[1])
        self.xqueue.pop(0) 

    def write_out(self, data, fi):
        '''
        Function to write out data

        @param data : data to write out
        @param fi : file to write to
        '''
        print(data)
        # Open and write
        with open(fi, "w", newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(data)