import argparse
import numpy as np
import os
import pickle
import sys


def read_lattices_edges(dataset_dir):
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

def main(args):
    """ Primary entry point for the script. """
    dataset_dir = args.target_dir
    if not os.path.isdir(dataset_dir):
        raise Exception('{} is not a valid directory'.format(dataset_dir))

    edge_count_list = read_lattices_edges(dataset_dir)
    stats_dict = generate_statistics(edge_count_list)
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
        '-i', '--target-dir', type=str,
        help='Location of the dataset with the uncompressed lattice files (*.npz)'
    )
    parser.add_argument(
        '-o', '--output-stats', type=str,
        help='The file to save the exploration statistics.'
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
