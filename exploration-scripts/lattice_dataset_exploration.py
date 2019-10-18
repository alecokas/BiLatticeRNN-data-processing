import argparse
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys


HEADER_LINE_COUNT = 8


def read_processed_lattices_edges(dataset_dir):
    """ For each lattice in the dataset, find the number of edges
        and return a list with the edge count for each lattice.
    """
    edge_count_list = []
    for root, _, names in os.walk(dataset_dir):
        for name in names:
            if name.endswith('.npz'):
                lattice_path = os.path.join(root, name)
                lattice = np.load(lattice_path)
                edge_count_list.append(len(lattice['edge_data']))
    return np.asarray(edge_count_list)

def generate_statistics(edge_count_list):
    """ Return a dictionary with some statistics """
    stats_dict = {}
    stats_dict['mean'] = np.mean(edge_count_list)
    stats_dict['std'] = np.std(edge_count_list)
    stats_dict['median'] = np.median(edge_count_list)
    stats_dict['pmf'] = np.histogram(edge_count_list, bins=np.arange(np.max(edge_count_list) + 1), density=True)
    return stats_dict

def save_results(results_dict, target_file):
    with open(target_file + '.pickle', 'wb') as tgt_file:
        pickle.dump(results_dict, tgt_file, protocol=pickle.HIGHEST_PROTOCOL)

def processed_lattice_exploration(args):
    """ Extract and return a list of the number of edges in each HTK lattice. """
    dataset_dir = args.lattice_dir
    if not os.path.isdir(dataset_dir):
        raise Exception('{} is not a valid directory'.format(dataset_dir))

    return read_processed_lattices_edges(dataset_dir)

def read_num_edges(lattice_path):
    """
    """
    with gzip.open(lattice_path, 'rt', encoding='utf-8') as file_in:
        for i in range(HEADER_LINE_COUNT):
            _ = file_in.readline()
        counts = file_in.readline().split()
        if len(counts) != 2 or counts[0].startswith('N=') is False or counts[1].startswith('L=') is False:
            raise Exception('Expected to find node and link information')
        else:
            num_nodes = float(counts[0].split('=')[1])
            num_edges = float(counts[1].split('=')[1])
            return num_nodes, num_edges

def read_htk_lattice_edges(base_dir, extention_dir):
    subsets = ['dev', 'eval']
    edge_count_list = []
    for subset in subsets:
        subset_dir = os.path.join(base_dir, subset)
        speaker_dirs = next(os.walk(os.path.join(base_dir, subset)))[1]
        for speaker_dir in speaker_dirs:
            raw_lattice_dir = os.path.join(subset_dir, speaker_dir, extention_dir)
            raw_lattice_list = next(os.walk(raw_lattice_dir))[2]
            for lattice_name in raw_lattice_list:
                abs_lat_path = os.path.join(raw_lattice_dir, lattice_name)
                _, num_edges = read_num_edges(abs_lat_path)
                edge_count_list.append(num_edges)
    return edge_count_list

def raw_lattice_exploration(args):
    """ """
    base_dir = args.base_lat_dir
    extention_dir = args.extension_dir
    return read_htk_lattice_edges(base_dir, extention_dir)

def count(array):
    assert array.ndim == 1
    array = array[array != np.array(None)]
    num_ones = np.sum(array)
    num_zeros = array.shape[0] - num_ones
    return num_zeros, num_ones

def target_counts(targets_path):
    targets_file = np.load(targets_path)
    # For all arcs
    targets = targets_file['target']
    num_zeros, num_ones = count(targets)
    # For one-best
    ref = targets_file['ref']
    onebest_zeros, onebest_ones = count(ref)
    # For competing arcs
    competing_zeros = num_zeros - onebest_zeros
    competing_ones = num_ones - onebest_ones
    return num_zeros, num_ones, onebest_zeros, onebest_ones, competing_zeros, competing_ones

def plot_distributions(array, file_name):
    """ Generate plots for the empirical probability mass distribution of the start, end, and duration times. """
    fig = plt.figure()
    n, bins, patches = plt.hist(x=array, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85, density=True)
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(ymax=np.max(n))
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Number of Occurrences', fontsize=16)
    plt.savefig(file_name, dpi=fig.dpi)

def dataset_balance(targets_dir):
    """ For each lattice in the dataset, find the number of edges
        tagged with a confidence of one and the number of edges tagged
        with a confidence of zero.
        TODO: This is quite lazy - could use functions to get rid of repetition
    """
    dataset_balance_dict = {
        'total-negative-tags': 0,
        'total-positive-tags': 0,
        'negative-tags-per-lattice': [],
        'positive-tags-per-lattice': []
    }

    onebest_balance_dict = {
        'onebest-negative-tags': 0,
        'onebest-positive-tags': 0,
        'negative-onebest-tags-per-lattice': [],
        'positive-onebest-tags-per-lattice': []
    }

    competing_balance_dict = {
        'competing-negative-tags': 0,
        'competing-positive-tags': 0,
        'negative-competing-tags-per-lattice': [],
        'positive-competing-tags-per-lattice': []
    }

    for root, _, names in os.walk(targets_dir):
        for name in names:
            if name.endswith('.npz'):
                target_path = os.path.join(root, name)
                zero_counts, one_counts, onebest_zeros, onebest_ones, competing_zeros, competing_ones = target_counts(target_path)
                # All counts
                dataset_balance_dict['total-negative-tags'] += zero_counts
                dataset_balance_dict['total-positive-tags'] += one_counts
                dataset_balance_dict['negative-tags-per-lattice'].append(zero_counts)
                dataset_balance_dict['positive-tags-per-lattice'].append(one_counts)

                # Onebest counts
                onebest_balance_dict['onebest-negative-tags'] += onebest_zeros
                onebest_balance_dict['onebest-positive-tags'] += onebest_ones
                onebest_balance_dict['negative-onebest-tags-per-lattice'].append(onebest_zeros)
                onebest_balance_dict['positive-onebest-tags-per-lattice'].append(onebest_ones)

                # Competing counts
                competing_balance_dict['competing-negative-tags'] += competing_zeros
                competing_balance_dict['competing-positive-tags'] += competing_ones
                competing_balance_dict['negative-competing-tags-per-lattice'].append(competing_zeros)
                competing_balance_dict['positive-competing-tags-per-lattice'].append(competing_ones)

    dataset_balance_dict['total-pmf-pos'] = np.histogram(
        dataset_balance_dict['positive-tags-per-lattice'],
        bins=np.arange(np.max(dataset_balance_dict['positive-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(dataset_balance_dict['positive-tags-per-lattice'], 'total-pmf-pos')

    dataset_balance_dict['total-pmf-neg'] = np.histogram(
        dataset_balance_dict['negative-tags-per-lattice'],
        bins=np.arange(np.max(dataset_balance_dict['negative-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(dataset_balance_dict['negative-tags-per-lattice'], 'total-pmf-neg')

    onebest_balance_dict['onebest-pmf-pos'] = np.histogram(
        onebest_balance_dict['positive-onebest-tags-per-lattice'],
        bins=np.arange(np.max(onebest_balance_dict['positive-onebest-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(onebest_balance_dict['positive-onebest-tags-per-lattice'], 'onebest-pmf-pos')

    onebest_balance_dict['onebest-pmf-neg'] = np.histogram(
        onebest_balance_dict['negative-onebest-tags-per-lattice'],
        bins=np.arange(np.max(onebest_balance_dict['negative-onebest-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(onebest_balance_dict['negative-onebest-tags-per-lattice'], 'onebest-pmf-neg')

    competing_balance_dict['competing-pmf-pos'] = np.histogram(
        competing_balance_dict['positive-competing-tags-per-lattice'],
        bins=np.arange(np.max(competing_balance_dict['positive-competing-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(competing_balance_dict['positive-competing-tags-per-lattice'], 'competing-pmf-pos')

    competing_balance_dict['competing-pmf-neg'] = np.histogram(
        competing_balance_dict['negative-competing-tags-per-lattice'],
        bins=np.arange(np.max(competing_balance_dict['negative-competing-tags-per-lattice']) + 1),
        density=True
    )
    plot_distributions(competing_balance_dict['negative-competing-tags-per-lattice'], 'competing-pmf-neg')
    return dataset_balance_dict, onebest_balance_dict, competing_balance_dict

def read_pickle(file_name):
    """ Load the pickle file
    """
    with (open(file_name, "rb")) as openfile:
        return pickle.load(openfile)

def visualise(stats, pickle_name):
    num_arcs_hist, num_arcs_bin_edges = stats['pmf']
    histogram_image(
        hist=num_arcs_hist,
        bin_edges=num_arcs_bin_edges,
        file_name='{}-{}'.format(pickle_name[:-7], 'arc-count-distribution.png')
    )
    # Total
    pos_count_hist, pos_count_bins = stats['total-pmf-pos']
    histogram_image(
        hist=pos_count_hist,
        bin_edges=pos_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'total-pos-distribution.png')
    )
    neg_count_hist, neg_count_bins = stats['total-pmf-neg']
    histogram_image(
        hist=neg_count_hist,
        bin_edges=neg_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'total-neg-distribution.png')
    )
    # One-best
    pos_count_hist, pos_count_bins = stats['onebest-pmf-pos']
    histogram_image(
        hist=pos_count_hist,
        bin_edges=pos_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'onebest-pos-distribution.png')
    )
    neg_count_hist, neg_count_bins = stats['onebest-pmf-neg']
    histogram_image(
        hist=neg_count_hist,
        bin_edges=neg_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'onebest-neg-distribution.png')
    )
    # Competing arcs
    pos_count_hist, pos_count_bins = stats['competing-pmf-pos']
    histogram_image(
        hist=pos_count_hist,
        bin_edges=pos_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'competing-pos-distribution.png')
    )
    neg_count_hist, neg_count_bins = stats['competing-pmf-neg']
    histogram_image(
        hist=neg_count_hist,
        bin_edges=neg_count_bins,
        file_name='{}-{}'.format(pickle_name[:-7], 'competing-neg-distribution.png')
    )

def histogram_image(hist, bin_edges, file_name):
    plt.bar(bin_edges[:-1], hist, width=1, color='r')
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.savefig(file_name)

def main(args):
    """ Primary entry point for the script. """
    if args.visualise_stats != 'None':
        stats = read_pickle(args.visualise_stats)
        visualise(stats, args.visualise_stats)
    else:
        if args.processed:
            edge_count_list = processed_lattice_exploration(args)
        else:
            edge_count_list = raw_lattice_exploration(args)

        stats_dict = generate_statistics(edge_count_list)

        if args.processed and args.dataset_balance:
            dataset_balance_dict, onebest_balance_dict, competing_balance_dict = dataset_balance(args.target_dir)
            stats_dict.update(dataset_balance_dict)
            stats_dict.update(onebest_balance_dict)
            stats_dict.update(competing_balance_dict)

        save_results(stats_dict, args.output_stats)

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.

        Arguments:
            args_to_parse:
    """
    description = "Do some data exploration of processed HTK lattices"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-o', '--output-stats', type=str, default='lattice-statistics',
        help='The file to save the exploration statistics.'
    )
    parser.add_argument(
        '--processed', default=False, action="store_true",
        help='Flag to indicate that the operation is operating on processed lattices'
    )
    parser.add_argument(
        '-l', '--lattice-dir', type=str, default='',
        help='Location of the dataset with the unzipped lattice files (*.npz)'
    )
    parser.add_argument(
        '-b', '--base-lat-dir', type=str, default='',
        help="Path to the base lattice directory which contains the dev and eval splits expected with BABEL."
    )
    parser.add_argument(
        '-e', '--extension-dir', type=str, default='',
        help="Extension directory post-dataset directory"
    )
    parser.add_argument(
        '--dataset-balance', default=False, action="store_true",
        help='Flag to indicate that the operation is operating on processed lattices'
    )
    parser.add_argument(
        '-t', '--target-dir', type=str, default='',
        help='Location of the targets which accompany the processed dataset (*.npz)'
    )
    parser.add_argument(
        '-v', '--visualise-stats', type=str, default='None',
        help='The name of the stats file to visualise'
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
