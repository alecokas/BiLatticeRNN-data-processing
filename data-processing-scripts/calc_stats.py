#!/usr/bin/env python3
""" Compute the statistics of the dataset (train and CV set) and save it in a file named stats.npz.
    For speed and accuracy on large dataset, the algorithm follows
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


WORD_EMBEDDING_LENGTH = 50
SUBWORD_EMBEDDING_LENGTH = 4


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
    word_aggregate = None
    subword_aggregate = None
    for data_file in args:
        with open(data_file, 'r') as file_in:
            iterator = tqdm(file_in.readlines(), dynamic_ncols=True)
            for line in iterator:
                lattice = np.load(line.strip())
                word_data = lattice['edge_data']
                if 'grapheme_data' in lattice:
                    grapheme_lattice = True
                    subword_data = lattice['grapheme_data']
                else:
                    grapheme_lattice = False
                    # A dummy variable to iterate since we do not have any actual subword data
                    subword_data = np.zeros_like(word_data)

                # Backward compatibility
                try:
                    ignore = list(lattice['ignore'])
                except KeyError:
                    ignore = []
                for i, (word_entry, subword_entry) in enumerate(zip(word_data, subword_data)):
                    if i not in ignore:
                        if word_aggregate is None:
                            # (count, mean, squared distance from mean)
                            word_aggregate = (1, word_entry, np.zeros(word_entry.shape[0]))
                            # sub-word initialisation and aggregation
                            if grapheme_lattice:
                                subword_aggregate = (1, subword_entry[0,:], np.zeros(subword_entry.shape[1]))
                                for subword in subword_entry[1:,:]:
                                    if not all(feature == 0 for feature in subword):
                                        subword_aggregate = update(subword_aggregate, subword)
                        else:
                            word_aggregate = update(word_aggregate, word_entry)
                            if grapheme_lattice:
                                for subword in subword_entry:
                                    if not all(feature == 0 for feature in subword):
                                        subword_aggregate = update(subword_aggregate, subword)

    word_mean, word_variance = finalize(word_aggregate)
    if grapheme_lattice:
        subword_mean, subword_variance = finalize(subword_aggregate)
    else:
        subword_mean = None
        subword_variance = None
    return word_mean, word_variance, subword_mean, subword_variance

def mask(mean, variance, mask_length):
    """ Mask the embedding with a mean of zero and variance of 1 so that
        whitening does not modify the embedding.
    """
    mask = range(0, mask_length)
    mean[mask] = 0.0
    variance[mask] = 1.0
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
        '-d', '--dest-dir', type=str, required=True,
        help='Full path to the directory in which to save the mean and variance stats.'
    )
    args = parser.parse_args()

    word_mean, word_variance, subword_mean, subword_variance = compute_stats(args.train_file, args.validation_file)
    stats_file = os.path.join(args.dest_dir, 'stats.npz')

    word_mean, word_variance = mask(word_mean, word_variance, WORD_EMBEDDING_LENGTH)

    if subword_mean is not None and subword_variance is not None:
        subword_mean, subword_variance = mask(subword_mean, subword_variance, SUBWORD_EMBEDDING_LENGTH)
        np.savez(
            stats_file,
            mean=np.reshape(word_mean, (1, -1)),
            std=np.reshape(np.sqrt(word_variance), (1, -1)),
            subword_mean=np.reshape(subword_mean, (1, -1)),
            subword_std=np.reshape(np.sqrt(subword_variance), (1, -1))
        )
    else:
        np.savez(stats_file, mean=np.reshape(word_mean, (1, -1)), std=np.reshape(np.sqrt(word_variance), (1, -1)))

if __name__ == '__main__':
    main()
