"""This module contains standalone utility functions that can be used across
other modules.
"""
import numpy as np

def get_n_ranges(start, end, num_ranges):
    """Takes a start index, end index, and number of ranges, e.g. start=0,
    end=2 and num_ranges=2, and returns a list of lists indicating
    evenly-spaced ranges, i.e.  [[0, 1], [1, 2]].
    """
    spacing = np.linspace(start, end, num_ranges + 1).astype(np.int)

    ranges = []
    for spacing_index in range(len(spacing) - 1):
        ranges.append([spacing[spacing_index], spacing[spacing_index + 1]])

    return ranges
