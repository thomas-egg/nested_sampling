from nested_sampling import MCWalker_mcpele
from mcpele.monte_carlo import Metropolis_MCrunner
from pele.potentials import LJ
import numpy as np

if __name__ == '__main__':

    start_coords = np.array([1.,1.,1.,1.,1.,-1.,1.,-1.,1.,-1.,1.,1.,1.,-1.,-1.,-1.,1.,-1.])
    temperature = 0.2
    pot = LJ()
    niter = 100
    stepsize = 0.5
    print(pot.getEnergy(start_coords))
    mcrunner = Metropolis_MCrunner(
        pot,
        start_coords,
        temperature,
        stepsize,
        niter,
        hEmin=-140,
        adjustf=0.9,
        adjustf_niter=10,
    )

    mcrunner.run()
    print(mcrunner.get_coords())
    print(mcrunner.get_success())
    print(mcrunner.get_accepted_fraction())
    print(mcrunner.get_iterations_count())
    print(mcrunner.get_neval())


