# Import libraries
from pele.potentials import LJCut
from nested_sampling import NestedSampling, MCWalker_mcpele, Replica
import argparse

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
    parser.add_argument('nlive', metavar='K', type=float, help='Number of live particles')
    parser.add_argument('nproc', metavar='N', type=int, help='Number of processors')
    parser.add_argument('nsteps', metavar='S', type=int, help='Number of steps')
    parser.add_argument('nparticles', metavar='P', type=int, help='Number of particles')
    parser.add_argument('density', metavar='D', type=float, help='Target density') 
    parser.add_argument('sigma', metavar='sig', type=float, help='LJ sigma')
    parser.add_argument('eps', metavar='eps', type=float, help='LJ epsilon')  
    parser.add_argument('ndim', metavar='dim', type=float, help='Dimensionality')   
    args = parser.parse_args()

    # Variable init
    nlive = args.nlive
    nparticles = args.nparticles
    density = args.density
    sigma = args.sigma
    eps = args.eps
    ndim = args.ndim
    box = get_boxvec(density, nparticles, sigma)

    # Instantiate potential
    # Assuming sigma = 1 and eps = 1
    pot = LJCut(eps, sigma, rcut=3.0*sigma, boxvec=box)
    replicas = [Replica(x, pot.get_energy(x)) for x in [np.random.uniform(low=0, high=box[0], size=(nparticles, ndim)) for _ in range(npoints)]]
    mc = MCWalker_mcpele(pot)
    sampler = NestedSampling(replicas, mc, cpfreq=500, iprint=500, cpfile='chk.txt', use_mcpele=True, sampler=10)
    sampler.run_sampler(3.0*(10 ** 12))