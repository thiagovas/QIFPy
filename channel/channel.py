"""
  TODO(thiagovas): Explain a bit about the channel class.
"""

__all__ = ["Channel"]


def BaseDistribution(object):
  """Probability Distribution.

    Utility class which represents probability distributions.
    It also contains utilitary functions, such as IsDistribution,
    that checks whether a given array represents a probability distribution.
    Further, it contains information theoretic functions, as Shannon Entropy,
    Renyi min-entropy, etc.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

  def __init__(self):
    """Inits SampleClass with blah."""
    pass

  @staticmethod
  def IsDistribution(dist):
    """Fetches rows from a Bigtable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Args:
        big_table: An open Bigtable Table instance.
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {'Serak': ('Rigel VII', 'Preparer'),
         'Zim': ('Irk', 'Invader'),
         'Lrrr': ('Omicron Persei 8', 'Emperor')}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    pass

  def MutualInformation(self):
    pass

  def NormalizedMutualInformation(self):
    pass

  def ShannonEntropy(self):
    pass

  def RenyiEntropy(self):
    pass

  def GuessingEntropy(self):
    pass






def Channel(object):
  def __init__(self):
    pass
