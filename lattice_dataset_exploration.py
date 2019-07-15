import argparse
import gzip
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


def target_counts(lattice_path):
    lattice = np.load(lattice_path)
    targets = lattice['target']
    assert targets.ndim == 1
    num_ones = np.sum(targets)
    num_zeros = targets.shape[0] - num_ones
    return num_zeros, num_ones


def dataset_balance(dataset_dir):
    """ For each lattice in the dataset, find the number of edges
        tagged with a confidence of one and the number of edges tagged
        with a confidence of zero.
    """
    dataset_balance_dict = {
        'false-tags': 0,
        'positive-tags': 0
    }
    for root, _, names in os.walk(dataset_dir):
        for name in names:
            if name.endswith('.npz'):
                lattice_path = os.path.join(root, name)
                zero_counts, one_counts = target_counts(lattice_path)
                dataset_balance_dict['false-tags'] += zero_counts
                dataset_balance_dict['positive-tags'] += one_counts
    return dataset_balance_dict


def read_pickle(file_name):
    """ Load the pickle file
    """
    with (open(file_name, "rb")) as openfile:
        return pickle.load(openfile)


def visualise(stats, pickle_name):
    print(stats)
    hist, bin_edges = stats['pmf']
    plt.bar(bin_edges[:-1], hist, width = 1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.savefig('{}-{}'.format(pickle_name, 'arc-distribution.png'))


def main(args):
    """ Primary entry point for the script. """
    if args.visualise_stats != '':
        stats = read_pickle(args.visualise_stats)
        visualise(stats, args.visualise_stats)
    else:
        if args.processed:
            edge_count_list = processed_lattice_exploration(args)
        else:
            edge_count_list = raw_lattice_exploration(args)

        stats_dict = generate_statistics(edge_count_list)

        if args.processed and args.dataset_balance:
            dataset_balance_dict = dataset_balance(args.target_dir)
            stats_dict.update(dataset_balance_dict)

        save_results(stats_dict, args.output_stats)
        print(stats_dict)



def parse_arguments(args_to_parse):
    """ Parse the command line arguments.

        Arguments:
            args_to_parse:
    """
    description = "Do some data exploration of processed HTK lattices"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-o', '--output-stats', type=str, required=True,
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
