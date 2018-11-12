"""
  TODO(thiagovas): Explain a bit about the channel class.
"""

from typing import Iterable
from typing import List
from math import isclose
from numpy.random import uniform as numpy_uniform
from scipy.stats import entropy as scipy_entropy

__all__ = ["BaseDistribution", "Channel"]


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
    def IsDistribution(dist: Iterable[float]) -> bool:
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

    def Randomize(self) -> None:
        """Randomize the current probability distribution."""
        dist_sum = 0.0
        self._dist = []
        for x in range(self._dist_size):
            new_p = numpy_uniform()
            self._dist.append(new_p)
            dist_sum += new_p

        for i in range(len(self._dist)):
            self._dist[i] /= dist_sum


    def ShannonEntropy(self, base: float = 2) -> float:
        """Calculates the Shannon entropy.
        
        Args:
            base: The logarithmic base to use, defaults to 2.
        """
        return scipy_entropy(self._dist, base)

    def RenyiMinEntropy(self) -> float:
        raise NotImplementedError("Renyi min-entropy not"
                                  "implemented yet")

    def GuessingEntropy(self) -> float:
        """Calculates the Guessing entropy."""
        tmp_dist = reverse(sorted(self._dist))
        gentropy = 0.0
        question_index = 1
        for x in tmp_dist:
            gentropy += question_index*x
            question_index += 1
        return gentropy
        

class Channel(object):
    def __init__(self):
        pass
