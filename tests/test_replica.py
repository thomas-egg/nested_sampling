# Import Replica
from nested_sampling import Replica
import unittest

class TestReplica(unittest.TestCase):

    def setUp(self):

        # Set up replica
        x = [1.0, 0.7]
        E = 20
        self.rep = Replica(x, E)

    def test_getter(self):

        # Assert
        pos = self.rep.x
        self.assertAlmostEqual(pos[0], 1.0)
        self.assertAlmostEqual(pos[1], 0.7)