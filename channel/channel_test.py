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
        self.assertTrue(BaseDistribution.is_distribution([0.25, 0.25, 0.25, 0.25]))
        self.assertTrue(BaseDistribution.is_distribution([0.5, 0.5]))
        self.assertTrue(BaseDistribution.is_distribution([0, 0, 0, 1]))

        for i in range(1, 10):
            dist = BaseDistribution(n_items = i)
            self.assertTrue(BaseDistribution.is_distribution(dist._dist))
    
    def test_IsDistributionFalse(self):
        """
            Tests whether the IsDistribution function returns false when it is
            supposed to.
        """
        self.assertFalse(BaseDistribution.is_distribution([0.75, 0.2, 0.3]))
        self.assertFalse(BaseDistribution.is_distribution([-0.5, 0.5]))
        self.assertFalse(BaseDistribution.is_distribution([0.2, 0.2, 0.2]))
        self.assertFalse(BaseDistribution.is_distribution([0.2, -0.2, 0.5, 0.5]))

    def test_RandomizeTrue(self):
        """
            Tests whether the Randomize function works the way it is supposed
            to.
        """
        for i in range(1, 15):
            dist = BaseDistribution(n_items=i)
            old_dist = deepcopy(dist._dist)
            dist.randomize()
            self.assertTrue(BaseDistribution.is_distribution(dist._dist))
            if i > 4:
                self.assertNotEqual(old_dist, dist._dist)

    def test_RenyiMinEntropy(self):
        """
            Tests whether the RenyiMinEntropy function works the way it is
            supposed to.
        """
        dist = BaseDistribution(n_items=4)
        self.assertEqual(dist.renyi_min_entropy(base=2), -2)

    def test_ShannonEntropy(self):
        """
            Tests whether the ShannonEntropy function works the way it is
            supposed to.
        """
        dist = BaseDistribution(n_items=1)
        self.assertEqual(dist.shannon_entropy(base=2), 0)
        
        dist = BaseDistribution(n_items=2)
        self.assertEqual(dist.shannon_entropy(base=2), 1)
        
        dist = BaseDistribution(n_items=4)
        self.assertEqual(dist.shannon_entropy(base=2), 2)
        
        dist = BaseDistribution(n_items=8)
        self.assertEqual(dist.shannon_entropy(base=2), 3)

    def test_GuessingEntropy(self):
        pass


class ChannelTest(TestCase):
    """
        Class with tests done over the Channel class.
    """

    def test_IsDistributionTrue(self):
        """
        """
        pass


if __name__ == '__main__':
    unittest_main()
