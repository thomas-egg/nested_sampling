# Import libraries
import numpy as np
from pele.potentials import LJ
from nested_sampling import NestedSampling, MCWalker_mcpele, Replica
import argparse
import inspect

########
# MAIN #
########

def get_boxvec(density, nparticles, sigma):
    '''
    Function to compute box vector from target density

    @param density : target density
    @param nparticles : number of particles
    @param sigma : LJ param
    @return boxvec : vector of box lengths
    '''

    # Compute target length
    vec_size = (nparticles / density) ** (1/3)
    boxvec = [vec_size * sigma] * 3

    # Return 
    return boxvec

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Nested Sampling on LJ-31')
    parser.add_argument('-K', metavar='--nlive', type=int, help='Number of live points')
    parser.add_argument('-N', metavar='--nproc', type=int, help='Number of processors')
    parser.add_argument('-S', metavar='--nsteps', type=int, help='Number of steps')
    parser.add_argument('-P', metavar='--nparticles', type=int, help='Number of particles')
    parser.add_argument('-D', metavar='--density', type=float, help='Target density') 
    parser.add_argument('-sig', metavar='--sigma', type=float, help='LJ sigma')
    parser.add_argument('-eps', metavar='--eps', type=float, help='LJ epsilon')  
    parser.add_argument('-dim', metavar='--ndim', type=int, help='Dimensionality')   
    args = parser.parse_args()
    
    if None in (args.K, args.N, args.P, args.D, args.sig, args.eps, args.dim):
        parser.error("Please provide all required arguments.")

    # Variable init
    nlive = args.K
    nproc = args.N
    nparticles = args.P
    density = args.D
    sigma = args.sig
    eps = args.eps
    ndim = args.dim
    steps = args.S
    box = get_boxvec(density, nparticles, sigma)

    # Instantiate potential
    # Assuming sigma = 1 and eps = 1
    pot = LJ(eps, sigma, boxvec=box)
    replicas = [Replica(x, pot.getEnergy(x)) for x in [np.random.uniform(low=0, high=box[0], size=(nparticles*ndim)) for _ in range(nlive)]]
    sampler = NestedSampling(replicas, pot, temp=1, nproc=nproc, cpfreq=1000, verbose=False, iprint=1000, chkpt=True, enfile='en_5_new.txt', cpfile='chk_5_new.txt', use_mcpele=False)
    sampler.run_sampler(steps)
