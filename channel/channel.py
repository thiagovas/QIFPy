"""
    File with the Channel class and its helper classes.
"""

from math import isclose
from math import log
from numpy.random import uniform as numpy_uniform
from scipy.stats import entropy as scipy_entropy
from typing import Iterable
from typing import List

__all__ = ["BaseDistribution", "Channel", "ChannelOptions"]


class BaseDistribution(object):
    """Probability Distribution.

      Utility class which represents probability distributions.
      It also contains utilitary functions, such as IsDistribution,
      that checks whether a given array represents a probability distribution.
      Further, it contains information theoretic functions, as Shannon Entropy,
      Renyi min-entropy, etc.

      Attributes:
          None.
    """

    def __init__(self, n_items: int = 1, dist: List[float] = None) -> None:
        """Inits BaseDistribution with a uniform distribution of size
            n_items.

            One can also build an instance of this class from a previous
            distribution by setting the dist attribute.

          Attributes:
              n_items: The number of entries of the probability distribution.
              dist: A vector of floats representing a probability distribution.
        """
        if dist:
            self._dist = dist
            self._dist_size = len(dist)
        else:
            self._dist = [1.0/n_items for x in range(n_items)]
            self._dist_size = n_items

    @staticmethod
    def is_distribution(dist: Iterable[float]) -> bool:
        """Returns whether a given array represents a distribution or not.

        The function checks whether there is no negative numbers at the input,
        and whether the sum of all values equals to 1.

        Args:
            dist: An open Bigtable Table instance.

        Returns:
            A boolean, true if the parameter represents a probability
            distribution, and false otherwise.

        Raises:
            Nothing.
        """
        dist_sum = 0.0
        for x in dist:
            if x < 0:
                return False
            else:
                dist_sum += x
        return isclose(dist_sum, 1.0, rel_tol=1e-6)

    def randomize(self) -> None:
        """Randomize the current probability distribution.
        
        Args:
            None.
        
        Returns:
            Nothing.
        """
        dist_sum = 0.0
        self._dist = []
        for x in range(self._dist_size):
            new_p = numpy_uniform()
            self._dist.append(new_p)
            dist_sum += new_p

        for i in range(len(self._dist)):
            self._dist[i] /= dist_sum


    def shannon_entropy(self, base: float = 2) -> float:
        """Calculates the Shannon entropy.
        
        Args:
            base: The logarithmic base to use, defaults to 2.

        Returns:
            A float value, the shannon entropy of the distribution.
        """
        return scipy_entropy(self._dist, base=base)
    
    def bayes_entropy(self) -> float:
        """
        """
        pass    

    def renyi_min_entropy(self, base: float = 2) -> float:
        """Calculates the Renyi min-entropy.

        Args:
            base: The logarithmic base to use, defaults to 2.

        Returns:
            A float value, the Renyi Min-Entropy of the distribution.
        """
      
        entropy = 0.0
        for p in self._dist:
            entropy = max(entropy, p)
        return log(entropy, base)

    def guessing_entropy(self) -> float:
        """Calculates the Guessing entropy.
    
        Args:
            None.

        Returns:
            A float value, the guessing entropy of the distribution.
        """
        tmp_dist = reverse(sorted(self._dist))
        gentropy = 0.0
        question_index = 1
        for x in tmp_dist:
            gentropy += question_index*x
            question_index += 1
        return gentropy


class ChannelOptions(object):
    """Class with options set to Channel.
       
       By default the options are set to a 2x2 identity channel.

        Attributes:
        c_matrix: An array of BaseDistribution objects representing the
                    conditional matrix of a channel.
        prior: A BaseDistribution object representing the prior
                distribution of the channel.    
    """

    def __init__(self):
        """Constructor for Channel Options."""
        self.SetIdentity(2)
        self.c_matrix = [BaseDistribution(dist=[1, 0]),
                         BaseDistribution(dist=[0, 1])]
        self.prior = BaseDistribution(dist=[0.5, 0.5])

    def set_c_matrix(self, matrix: Iterable[BaseDistribution]) -> ChannelOptions:
        """ """
        self.c_matrix = matrix

    def set_prior_distribution(self, dist: BaseDistribution) -> ChannelOptions:
        """ """
        seld.prior = dist

    def set_identity(self, dimension: int = 2) -> ChannelOptions:
        """ """
        self.c_matrix = []
        prior_dist = [1.0/dimension for d in range(dimension)]

        for d in range(dimension):
            c_dist = [0.0]*dimension
            c_dist[d] = 1.0
            self.c_matrix.append(BaseDistribution(dist=c_dist))


class Channel(object):
    """Information Theoretic Channel.

      Class representing a information theoreric channel. The class is
      instantiated through a set of options defined at the ChannelOptions
      class.

      Attributes:
        options: Instance of ChannelOptions, describing the channel with the
                 conditional distributions and prior.
        outter: Instance of BaseDistribution containing the outter distribution
                of the channel.
    """

    def __init__(self, options: ChannelOptions):
        self.options = options

    def __str__(self):
        pass

    def randomize(self):
        pass

    def compute_j_matrix(self) -> None:
        pass

    def compute_outter_distribution(self) -> None:
        pass

    def compute_max_prior(self) -> None:
        pass

    def compute_max_poutter(self) -> None:
        pass

    def mutual_information(self) -> float:
        pass

    def normalized_mutual_information(self) -> float:
        pass

    def symmetric_uncertainty(self) -> float:
        pass

    def conditional_entropy(self) -> float:
        pass

    def conditional_entropy_hyper(self) -> float:
        pass

    def joint_shannon_entropy(self) -> float:
        pass

    def joint_guessing_entropy(self) -> float:
        pass
