# Import libraries
import numpy as np
#from pele.potentials import LJCut
from nested_sampling import NestedSampling, MCWalker_mcpele, Replica
import argparse
import inspect
from numba import jit

@jit(nopython=True)
def apply_pbc(rij, box_length):
    """Apply minimum image convention to account for periodic boundary conditions."""
    return rij - box_length * np.round(rij / box_length)

class LJCut(object):
    def __init__(self, eps, sig, rcut, bv):
        self.eps = eps
        self.sig = sig
        self.rcut = rcut
        self.bv = bv

    @jit(nopython=True)
    def __call__(self, positions):
        """Compute Lennard-Jones potential with cutoff and periodic boundary conditions (PBC)."""
        energy = 0.0
        n_atoms = len(positions)
        cutoff_sq = self.rcut ** 2  # Square the cutoff for efficiency

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Distance vector between atoms i and j
                rij = positions[i] - positions[j]
                # Apply periodic boundary conditions
                rij = apply_pbc(rij, self.bv[0])
                # Squared distance
                r_sq = np.dot(rij, rij)
    
                # Only compute if within cutoff distance
                if r_sq < cutoff_sq:
                    # Compute r^2 to r^6 and r^12
                    r2_inv = self.sig ** 2 / r_sq
                    r6_inv = r2_inv ** 3
                    r12_inv = r6_inv ** 2
                    # Lennard-Jones potential
                    energy += 4 * self.eps * (r12_inv - r6_inv)

        return energy


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
    pot = LJCut(eps, sigma, rcut=3*sigma, boxvec=box)
    replicas = [Replica(x, pot.getEnergy(x)) for x in [np.random.uniform(low=0, high=box[0], size=(nparticles*ndim)) for _ in range(nlive)]]
    sampler = NestedSampling(replicas, pot, temp=1, nproc=nproc, cpfreq=1000, verbose=False, iprint=1000, chkpt=True, enfile='serial/en5.txt', cpfile='serial/chk5.txt', use_mcpele=True, takestep=None)
    sampler.run_sampler(steps)
