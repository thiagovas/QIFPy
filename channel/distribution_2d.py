"""
    Distribution2D contains the representation of a probability distribution
"""

from copy import deepcopy
from math import isclose
from math import log
from numpy.random import uniform as numpy_uniform
from scipy.stats import entropy as scipy_entropy
from typing import Iterable
from typing import List

__all__ = ["Distribution2D"]


class Distribution2D(BaseDistribution):
    """Helper class representing a two dimensional probability distribution.

       The class contains helper methods to convert indices, and specific
       initializers.
    """

    def __init__(self, n_rows: int = 2, n_columns: int = 2):
        """Constructor of the Distribution2D class.

        Args:
            n_rows: An integer value, the number of rows of the distribution.
            n_columns: An integer value, the number of columns of the
                       distribution.
        """
        self._n_rows = n_rows
        self._n_columns = n_columns
        super().__init__(n_items = n_rows * n_columns)

    def _get_linear_index(self, row_index: int, column_index: int) -> int:
        """Converts 2D indices to a 1D index.

            All indices are 0-based.

        Args:
            row_index: An integer value, the index of the row of the
                       distribution.
            column_index: An integer value, the index of the column of the
                          distribution.

        Returns:
            An integer value, the correspondent index on a 1D array which
            corresponds the 2D indices parameters.
        """
        return self._n_columns*row_index + column_index

    def get_p(self, row_index: int, column_index: int) -> float:
        """Getter of the probability.

        Args:
            row_index: An integer value, the index of the row of the
                       distribution.
            column_index: An integer value, the index of the column of the
                          distribution.

        Returns:
            A float value, the probability p(row_index, column_index).
        """
        index_dist = self._get_linear_index(row_index, column_index)
        return self._dist[index_dist]

    def set_p(self, row_index: int, column_index: int, value: float) -> None:
        """Setter to a cell of the probability distribution.

        Args:
            row_index: An integer value, the index of the row of the
                       distribution.
            column_index: An integer value, the index of the column of the
                          distribution.
            value: A float value, the value to which the cell will assume.

        Returns:
            Nothing.
        """
        index_dist = self._get_linear_index(row_index, column_index)
        self._dist[index_dist] = value

