"""
  File that contains unit tests and test fixtures to evaluate the correctness
  of the classes found at the channel.py file.
"""

from unittest import TestCase
from unittest import main as unittest_main
from copy import deepcopy
from channel import BaseDistribution

class BaseDistributionTest(TestCase):
    """
        Class with tests done over the BaseDistribution class.
    """

    def test_IsDistributionTrue(self):
        """
            Tests whether the IsDistribution function returns true when it is
            supposed to.
        """
        self.assertTrue(BaseDistribution.IsDistribution([0.25, 0.25, 0.25, 0.25]))
        self.assertTrue(BaseDistribution.IsDistribution([0.5, 0.5]))
        self.assertTrue(BaseDistribution.IsDistribution([0, 0, 0, 1]))

        for i in range(1, 10):
            dist = BaseDistribution(n_items = i)
            self.assertTrue(BaseDistribution.IsDistribution(dist._dist))
    
    def test_IsDistributionFalse(self):
        """
            Tests whether the IsDistribution function returns false when it is
            supposed to.
        """
        self.assertFalse(BaseDistribution.IsDistribution([0.75, 0.2, 0.3]))
        self.assertFalse(BaseDistribution.IsDistribution([-0.5, 0.5]))
        self.assertFalse(BaseDistribution.IsDistribution([0.2, 0.2, 0.2]))
        self.assertFalse(BaseDistribution.IsDistribution([0.2, -0.2, 0.5, 0.5]))

    def test_RandomizeTrue(self):
        """
            Tests whether the Randomize function works the way it is supposed
            to.
        """
        for i in range(1, 15):
            dist = BaseDistribution(n_items=i)
            old_dist = deepcopy(dist._dist)
            dist.Randomize()
            self.assertTrue(BaseDistribution.IsDistribution(dist._dist))
            if i > 4:
                self.assertNotEqual(old_dist, dist._dist)

    def test_RenyiMinEntropy(self):
        """
            Tests whether the RenyiMinEntropy function works the way it is
            supposed to.
        """
        with self.assertRaises(NotImplementedError):
            dist = BaseDistribution(n_items=3)
            dist.RenyiMinEntropy()

    def test_ShannonEntropy(self):
        pass

    def test_GuessingEntropy(self):
        pass


class ChannelTest(TestCase):
    """
        Class with tests done over the Channel class.
    """

    def test_IsDistributionTrue(self):
        """
        """


if __name__ == '__main__':
    unittest_main()
