#!/usr/bin/env python3
"""Compute the statistics of the dataset.
For speed and accuracy on large dataset, the algorithm follows
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

def update(cur_aggregate, new_value):
    """For a new value new_value, compute the new count, new mean, the new m_2.
    * mean accumulates the mean of the entire dataset.
    * m_2 aggregates the squared distance from the mean.
    * count aggregates the number of samples seen so far.
    """
    (count, mean, m_2) = cur_aggregate
    count = count + 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    m_2 = m_2 + delta * delta2
    return (count, mean, m_2)

def finalize(cur_aggregate):
    """Retrieve the mean and variance from an aggregate."""
    (count, mean, m_2) = cur_aggregate
    mean, variance = mean, m_2 / (count - 1)
    if count < 2:
        return float('nan')
    else:
        return mean, variance

def compute_stats(*args):
    """Compute the statistics across entire dataset."""
    aggregate = None
    for data_file in args:
        with open(data_file, 'r') as file_in:
            iterator = tqdm(file_in.readlines(), dynamic_ncols=True)
            for line in iterator:
                lattice = np.load(line.strip())
                data = lattice['edge_data']
                # Backward compatibility
                try:
                    ignore = list(lattice['ignore'])
                except KeyError:
                    ignore = []
                for i, entry in enumerate(data):
                    if i not in ignore:
                        if aggregate is None:
                            aggregate = (1, entry, np.zeros(entry.shape[0]))
                        else:
                            aggregate = update(aggregate, entry)
    mean, variance = finalize(aggregate)
    return mean, variance

def main():
    """Main function for computing the statistics."""
    parser = argparse.ArgumentParser(description='generate lattice paths')
    parser.add_argument(
        '-t', '--train-file', type=str, required=True, 
        help='Full path to a text file with the training set lattices'
    )
    parser.add_argument(
        '-v', '--validation-file', type=str, required=True, 
        help='Full path to a text file with the cross validation set lattices'
    )
    parser.add_argument(
        '-d', '--dest-file', type=str, required=True, 
        help='Full path name of the file in which to save the mean and variance stats'
    )
    args = parser.parse_args()

    
    mean, variance = compute_stats(args.train_file, args.validation_file)
    mask = range(0, 50)
    mean[mask] = 0.0
    variance[mask] = 1.0
    np.savez(args.dest_file, mean=np.reshape(mean, (1, -1)), std=np.reshape(np.sqrt(variance), (1, -1)))

if __name__ == '__main__':
    main()
