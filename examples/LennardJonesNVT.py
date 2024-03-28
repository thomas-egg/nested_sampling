# Import libraries
from pele.potentials import LJCut
from nested_sampling import NestedSampling, MCWalker_mcpele, Replica
import argparse

########
# MAIN #
########
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Nested Sampling on LJ-31')
    parser.add_argument('nlive', metavar='N', type=float, help='Number of live particles')
    parser.add_argument('nproc', metavar='M', type=int, help='Number of processors') 
    args = parser.parse_args()

    # Instantiate potential
    # Assuming sigma = 1 and eps = 1
    nlive = args.nlive
    pot = LJCut(eps=1.0, sigma=1.0, rcut=3.0, boxvec=[23.7, 23.7, 23.7])
    replicas = [Replica(x, pot.get_energy(x)) for x in [np.random.uniform(low=0, high=23.7, size=(31, ndim)) for _ in range(npoints)]]
    mc = MCWalker_mcpele(pot)
    sampler = NestedSampling()