"""
    File with the Channel class and its helper classes.
"""

from copy import deepcopy
from math import isclose
from math import log
from numpy.random import uniform as numpy_uniform
from scipy.stats import entropy as scipy_entropy
from typing import Iterable
from typing import List
from distribution_2d import Distribution2D
from base_distribution import BaseDistribution

__all__ = ["Channel", "ChannelOptions"]


class ChannelOptions(object):
    """Class with options set to Channel.

       By default the options are set to a 2x2 identity channel.

       Attributes:
        n_rows: An integer, the number of rows of the Channel.
        n_columns: An integer, the number of columns of the Channel.
        c_matrix: An array of BaseDistribution objects representing the
                    conditional matrix of a channel.
        prior: A BaseDistribution object representing the prior
                distribution of the channel.
    """

    def __init__(self, n_rows: int = 2, n_columns: int = 2):
        """Constructor for Channel Options."""
        self.set_uniform(n_rows, n_columns)

    def set_c_matrix(self, matrix: Iterable[BaseDistribution]):
        """Setter to the conditional matrix of the channel.

        Args:
            matrix: An iterable of BaseDistribution, representing the
                    conditional probability distributions.

        Returns:
            self, the instance of ChannelOptions.
        """
        # TODO(thiagovas): Perform some checks, such as if all BaseDistribution
        #                  instances are of the same size.
        self.c_matrix = matrix
        return self

    def set_prior_distribution(self, dist: BaseDistribution):
        """Setter to the prior distribution of the channel.

        Args:
            dist: An instance of BaseDistribution, the prior distribution.

        Returns:
            self, the instance of ChannelOptions.
        """
        # TODO(thiagovas): Raise an exception if len(dist) != self.n_rows.
        self.prior = dist
        return self
    
    def set_uniform(self, n_rows: int = 2, n_columns: int = 2):
        """Sets the options to a uniform n_rows x n_columns channel.
            
        Args:
            n_rows: An integer, the number of rows of the Channel.
            n_columns: An integer, the number of columns of the Channel.

        Returns:
            self, the instance of ChannelOptions.
        """
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.c_matrix = [BaseDistribution(n_items = n_columns) for x in range(n_columns)]
        self.prior = BaseDistribution(n_items = n_rows)
        return self

    def set_identity(self, dimension: int = 2):
        """Sets the options to a identity quadratic channel.

        Args:
            dimension: An int value, the dimension of the channel.

        Returns:
            self, the instance of ChannelOptions.
        """
        self.c_matrix = []
        self.prior = BaseDistribution(dist=[1.0/dimension for d in range(dimension)])
        self.n_rows = dimension
        self.n_columns = dimension

        for d in range(dimension):
            c_dist = [0.0]*dimension
            c_dist[d] = 1.0
            self.c_matrix.append(BaseDistribution(dist=c_dist))
        return self


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
        """Constructor of the Channel class.

           The function instantiates the protected attributes of the class:
            * _j_matrix - An instance of Distribution2D, representing the joint
                          distribution calculated over the prior and
                          conditionals set at the channel options.
            * _outter_dist - An instance of BaseDistribution, representing the
                             outter distribution.
            * _max_prior - A float value, representing the maximum value of the
                           prior distribution.
            * _max_poutter - A float value, representing the maximum value of
                             the outter distribution.
        """
        self.options = options
        self._j_matrix = None
        self._outter_dist = None
        self._max_prior = None
        self._max_poutter = None
        self.compute_all()

    def __str__(self):
        """ """
        pass

    def randomize(self):
        """Randomizes the channel."""
        for d in self.options.c_matrix:
            d.randomize()

        self.options.prior.randomize()
        self.compute_all()

    def compute_all(self) -> None:
        """Call all pre-computation functions.

        Args:
            None.

        Returns:
            Nothing.
        """
        self.compute_j_matrix()
        self.compute_outter_distribution()
        self.compute_max_prior()
        self.compute_max_poutter()

    def compute_j_matrix(self) -> None:
        """ """
        self._j_matrix = Distribution2D(n_rows=self.options.n_rows,
                                        n_columns=self.options.n_columns)
        cur_row = 0
        for dist in self.options.c_matrix:
            for column_index in range(self.options.n_columns):
                self._j_matrix.set_p(row_index=cur_row, column_index=column_index,
                                     value=dist.get_p(column_index)*self.options.prior.get_p(cur_row))
            cur_rows += 1

    def compute_outter_distribution(self) -> None:
        """ """
        pass

    def compute_max_prior(self) -> None:
        """ """
        pass

    def compute_max_poutter(self) -> None:
        """ """
        pass

    def mutual_information(self) -> float:
        """ """
        pass

    def normalized_mutual_information(self) -> float:
        """ """
        pass

    def symmetric_uncertainty(self) -> float:
        """ """
        pass

    def conditional_entropy(self) -> float:
        """Computes the conditional shannon entropy H(Y|X).

        Args:
            None.

        Returns:
            A float value, the conditional entropy H(Y|X).
        """
        pass

    def conditional_entropy_hyper(self) -> float:
        """Computes the conditional shannon entropy on the hyper distribution.
           H(X|Y).

        Args:
            None.

        Returns:
            A float value, the conditional entropy H(X|Y).
        """
        pass

    def joint_shannon_entropy(self) -> float:
        """ """
        pass

    def joint_guessing_entropy(self) -> float:
        """ """
        pass
