import numpy as np
from diffusive_nested_sampling.level import Level

class Levels(object):
    """
    Object for storing a collection of levels, initialize with
    0-level
    """
    def __init__(self, max_J: int, L: float=10.0, C: int=10000):
        """
        Initialize Levels for DNS
        :param max_J:
            Integer value for maximum level
        :type max_J: int
        :param L:
            Backtracking for level weights
        :type L: float
        :param C:
            Confidence value 
        :type C: int
        """
        self.max_J = max_J
        self.current_max_J = 0  # Current maximum
        self.levels = [Level(0, np.log(0.0), prev=None)]    # levels are initialized with index, logL, and a previous logX if exists
        self.C = C 
        self.L = L

    def append(self, boundary: float):
        """
        Add a level to the list of levels
        :param boundary:
            new likelihood boundary
        :type boundary: float
        """
        assert self.current_max_J < self.max_J, f"Number of levels may not exceed {self.max_J}"
        prev_log_X = self.levels[self.current_max_J].get_log_X
        self.levels.append(Level(self.current_max_J, log_likelihood_boundary=boundary, prev=prev_log_X))
        self.current_max_J += 1

    def get_level(self, ind: int):
        """
        Return the specified level
        :param ind:
            index of level
        :return level 
            the level requested
        :rtype level: Level
        """
        assert ind <= self.current_max_J, f"Level requested exceeds {self.current_max_J}"
        level = self.levels[ind]
        return level
    
    def update_levels(self, new_visits: np.array, new_xadj: np.array, exceeds: np.array):
        """
        Update each level based on particle statistics
        :param new_visits:
            New visits to each level.
        :type new_visits: list
        :param new_visits_xadj:
            New visits that will contribute to x adjustment.
        :type new_visits_xadj: list
        :param exceeds:
            Number of visits in j whose likelihood exceeds that of j+1.
        :type exceeds: list
        """

        for i in range(self.current_max_J):
            self.levels[i].set_visits(new_visits[i], new_xadj[i], exceeds[i])
            if i < self.current_max_J:
                self.levels[i+1].set_log_X(self.levels[i].get_log_X, self.levels[i].xadj_visits, self.levels[i].exceeds, self.C)

    def get_acceptance_ratio(self, j: int, k: int, beta: float):
        """
        Return acceptance ratio for MCMC step
        :param j, k:
            States to go to/from.
        :type j, k: int
        :param pos:
            Particle position.
        type pos: float
        :param l:
            Backtracking value
        :type l: float
        :param beta:
            Adjustment exponent.
        :type beta: float
        :return a
            Acceptance rate.
        :rtype a: float
        """

        # Compute acceptance
        w_prime = self.levels[k].level_weight(self.current_max_J, self.L, self.max_J) - self.levels[k].get_log_X
        w = self.levels[j].level_weight(self.current_max_J, self.L, self.max_J) - self.levels[j].get_log_X
        a = np.exp(w_prime - w)
        # if self.current_max_J == self.max_J:
        #    visits_j, exp_j = self.levels[j].get_visits()
        #    visits_k, exp_k = self.levels[k].get_visits()
        #    a *= (((visits_j + self.C) / (exp_j + self.C)) / ((visits_k + self.C) / (exp_k + self.C))) ** beta

        # Return
        return a