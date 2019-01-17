"""
  File that contains unit tests and test fixtures to evaluate the correctness
  of the classes found at the channel.py file.
"""

from unittest import TestCase
from unittest import main as unittest_main
from copy import deepcopy
from base_distribution import BaseDistribution
from distribution_2d import Distribution2D
from channel import Channel
from channel import ChannelOptions

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


class Distribution2DTest(TestCase):
    """
        Class with tests done over the Distribution2D class.
    """

    def test_get_linear_index(self):
        """Tests whether the _get_linear_index function
            correctly converts the indices.
        """
        dist = Distribution2D(n_rows=3, n_columns=3)
        self.assertEqual(dist._get_linear_index(0, 0), 0)
        self.assertEqual(dist._get_linear_index(0, 1), 1)
        self.assertEqual(dist._get_linear_index(0, 2), 2)
        self.assertEqual(dist._get_linear_index(1, 0), 3)
        self.assertEqual(dist._get_linear_index(1, 1), 4)
        self.assertEqual(dist._get_linear_index(1, 2), 5)
        self.assertEqual(dist._get_linear_index(2, 0), 6)
        self.assertEqual(dist._get_linear_index(2, 1), 7)
        self.assertEqual(dist._get_linear_index(2, 2), 8)

    def test_init(self):
        """Tests whether the Distribution2D class is correctly
            initialized.
        """
        dist = Distribution2D(3, 3)
        self.assertEqual(dist._dist_size, 9)
        self.assertEqual(len(dist._dist), 9)
        self.assertEqual(dist._dist[0], 1.0/9)


class ChannelTest(TestCase):
    """
        Class with tests done over the Channel class.
    """

    def test_randomize(self):
        """Tests whether the channel is randomized and still the channel
            restrictions are satisfied.
        """
        dim = 3
        c = Channel(ChannelOptions().set_identity(dimension=dim))

        for test_number in range(30):
            c.randomize()
            for d in range(dim):
                self.assertTrue(BaseDistribution.is_distribution(c.options.c_matrix[d]._dist))
            self.assertTrue(BaseDistribution.is_distribution(c.options.prior._dist))

    def test_init(self):
        """ """
        pass

    def test_mutual_information(self):
        """ """
        pass

    def test_normalized_mutual_information(self):
        """ """
        pass

    def test_symmetric_uncertainty(self):
        """ """
        pass

    def test_conditional_entropy(self):
        """ """
        pass

    def test_conditional_entropy_hyper(self):
        """ """
        pass

    def test_joint_shannon_entropy(self):
        """ """
        pass

    def test_joint_guessing_entropy(self):
        """ """
        pass


if __name__ == '__main__':
    unittest_main()
