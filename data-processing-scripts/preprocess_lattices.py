""" Process lattices from the *.lat.gz format into the processed *.npz format.
    At the same time, produce the file list with paths to the processed files.
    TODO: Introduce functionality for decision tree mapping
"""
#!/usr/bin/env python3

import argparse
import gzip
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import numpy.ma as ma
import os
import re
import sys
from toposort import toposort_flatten
import utils


EMBEDDING_LENGTH = 50
HEADER_LINE_COUNT = 8


def read_lattice(lattice_path, subword_embedding=None, embed_apostrophe=False):
    """Read HTK lattices.

    Arguments:
        lattice_path {string} -- absolute path of a compressed lattice
            in `.lat.gz` format
        subword_embedding -- subword embeddings store as a dictionary

    Returns:
        nodes {list} -- indexed by nodeID, each element is [time, word]
        edges {list} -- indexed by edgeID, each element is [parentID, childID,
            AM_score, LM_score]
        dependency {dict} -- {childID: {parentID, ...}, ...}, used for
            topological sort
        child_2_parent {dict} -- {childID: {parentID: edgeID, ...}, ...},
            mapping from childID to parentID
        parent_2_child {dict} -- {parentID: {childID: edgeID, ...}, ...},
            mapping from parentID to childID
        grapheme_data {list of arrays} -- [arc1, arc2, arc3, ...] where arcX = (num_graphemes * num_features)
    """
    utils.check_file(lattice_path)
    nodes = []
    edges = []
    grapheme_data = []
    dependency = {}
    child_2_parent = {}
    parent_2_child = {}
    with gzip.open(lattice_path, 'rt', encoding='utf-8') as file_in:
        for i in range(HEADER_LINE_COUNT):
            _ = file_in.readline()
        counts = file_in.readline().split()
        if len(counts) != 2 or counts[0].startswith('N=') is False \
                            or counts[1].startswith('L=') is False:
            LOGGER.error('wrong lattice format')
            sys.exit(1)
        node_num = int(counts[0].split('=')[1])
        edge_num = int(counts[1].split('=')[1])
        for i in range(node_num):
            line = file_in.readline().split()
            assert int(line[0].split('=')[1]) == i, \
                   'wrong lattice format when reading nodes'
            time = float(line[1].split('=')[1])
            word = line[2].split('=')[1].replace(r"'", r"\'")
            nodes.append([time, word])
        for i in range(edge_num):
            line = file_in.readline().split()
            edge_id = int(line[0].split('=')[1])
            assert(edge_id == i), 'wrong lattice format when reading edges'
            parent = int(line[1].split('=')[1])
            child = int(line[2].split('=')[1])
            am_score = float(line[3].split('=')[1])
            lm_score = float(line[4].split('=')[1])

            if line[5].split('=')[0] == 'r':
                # Remove the prnounciation information if it is present
                del line[5]
            if line[5].split('=')[0] == 'd':
                # Extract a grapheme feature vector of dimensions: (num_graphemes, num_features)
                if subword_embedding is not None:
                    grapheme_feature_array = utils.get_grapheme_info(line[5].split('=')[1], subword_embedding, embed_apostrophe)
                    grapheme_data.append(grapheme_feature_array)
                post_idx = 6
            else:
                post_idx = 5

            if len(line) >= post_idx + 1:
                if line[post_idx].split('=')[0] == 'p':
                    # Expect posterior information
                    post = float(line[post_idx].split('=')[1])
                else:
                    raise Exception('This lattice ({}) has an unrecognised arc parameter sequence'.format(lattice_path))

                edges.append([parent, child, am_score, lm_score, post])
            else:
                edges.append([parent, child, am_score, lm_score])

            if child not in dependency:
                dependency[child] = {parent}
                child_2_parent[child] = {parent: edge_id}
            else:
                dependency[child].add(parent)
                child_2_parent[child][parent] = edge_id
            if parent not in parent_2_child:
                parent_2_child[parent] = {child: edge_id}
            else:
                parent_2_child[parent][child] = edge_id

    # go through the array now and put it in a big masked array so it is just ine simple numpy array (I, J, F)
    if subword_embedding is not None:
        max_grapheme_seq_length = utils.longest_grapheme_sequence(grapheme_data)
        padded_grapheme_data = np.empty((len(grapheme_data), max_grapheme_seq_length, utils.len_subword_features()))
        mask = np.empty_like(padded_grapheme_data, dtype=bool)

        for arc_num, grapheme_seq in enumerate(grapheme_data):
            padded_grapheme_data[arc_num, :, :], mask[arc_num, :, :] = utils.pad_subword_sequence(grapheme_seq, max_grapheme_seq_length)

        masked_grapheme_data = ma.masked_array(padded_grapheme_data, mask=mask, fill_value=-999999)

        return nodes, edges, dependency, child_2_parent, parent_2_child, masked_grapheme_data
    else:
        return nodes, edges, dependency, child_2_parent, parent_2_child

def process_one_lattice(lattice_path, dst_dir, wordvec, subword_embedding,
                        embed_apostrophe, uniform_subword_durations,
                        processed_file_list_path=None):
    """ Process and save a lattice into *.npz format

        Arguments:
            lattice_path: String containing the absolute path to lattices `.lat.gz`
            dst_dir: Absolute path to destination directory as a string
            wordvec: The word vector dictionary obtained by calling `load_wordvec`
            subword_embedding: Dictionary with subword embeddings
            embed_apostrophe: Boolean indicator of whether to embed
                              apostrophes separately.
    """
    name = lattice_path.split('/')[-1].split('.')[0] + '.npz'
    print('Processing {}'.format(name))
    try:
        LOGGER.info(name)
        name = os.path.join(dst_dir, name)
        if not os.path.isfile(name):
            nodes, edges, dependency, child_2_parent, parent_2_child, grapheme_data \
                = read_lattice(lattice_path, subword_embedding, embed_apostrophe)
            topo_order = toposort_flatten(dependency)

            # for each edge, the information contains
            # [EMBEDDING_LENGTH, duration(1), AM(1), LM(1), arc_posterior(1)]
            edge_data = np.empty((len(edges), EMBEDDING_LENGTH + 1 + 1 + 1 + 1))
            start_times = []
            ignore = []
            for i, edge in enumerate(edges):
                start_node = edge[0]
                start_times.append(nodes[start_node][0])
                end_node = edge[1]
                time = nodes[end_node][0] - nodes[start_node][0]
                word = nodes[end_node][1]

                if word in wordvec:
                    edge_data[i] = np.concatenate(
                        (wordvec[word], np.array([time, edge[2], edge[3], edge[4]])), axis=0)
                else:
                    edge_data[i] = np.concatenate(
                        (np.zeros(EMBEDDING_LENGTH), np.array([time, edge[2], edge[3], edge[4]])), axis=0)
                    LOGGER.info('OOV word: {}\n'.format(word))
                    utils.append_to_file(word, 'oov.txt')

                if word in ['<s>', '</s>', '!NULL', '<hes>']:
                    ignore.append(i)

            # save multiple variables into one .npz file
            np.savez(name, topo_order=topo_order, child_2_parent=child_2_parent,
                     parent_2_child=parent_2_child, edge_data=edge_data,
                     ignore=ignore, grapheme_data=grapheme_data, start_times=start_times
            )
            if processed_file_list_path is not None:
                utils.append_to_file(os.path.abspath(name), processed_file_list_path)
    except OSError as exception:
        LOGGER.info('%s\n' %lattice_path + str(exception))

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
        TODO: Introduce functionality for decision tree mapping

        Arguments:
            args_to_parse: CLI arguments to parse
    """
    parser = argparse.ArgumentParser(description='Process lattices from HTK format to a condensed numpy format')

    parser.add_argument(
        '-d', '--dst-dir', type=str,
        help='Location to save the processed lattice files (*.npz)'
    )
    parser.add_argument(
        '-e', '--embedding', type=str, required=True,
        help='Full path to the file containing a dictionary with the grapheme / phone embeddings'
    )
    parser.add_argument(
        '-w', '--wordvec', type=str, required=True,
        help='Full path to the file containing a dictionary with the word vector embeddings'
    )
    parser.add_argument(
        '-f', '--file-list-dir', type=str,
        help='The directory containing the files with the lists of lattice absolute paths for each subset (*.lat.txt)'
    )
    parser.add_argument(
        '-p', '--processed-file-list-dir', type=str,
        help='The directory in which to save files with paths to the processed lattices (*.txt).'
    )
    # TODO
    # parser.add_argument('--uniform-subword-durations', dest='uniform_subword_durations', action='store_true')
    # parser.set_defaults(uniform_subword_durations=False)
    parser.add_argument('--embed-apostrophe', dest='embed_apostrophe', action='store_true')
    parser.set_defaults(embed_apostrophe=False)
    parser.add_argument(
        '-v', '--verbose',
        help='Set logging level: ERROR (default), WARNING (-v), INFO (-vv), DEBUG (-vvv)',
        action='count', default=0
    )
    parser.add_argument(
        '-n', '--num-threads',
        help='The number of threads to use for concurrency',
        type=int, default=30
    )
    args = parser.parse_args()
    return args

def main(args):
    """Main function for lattice preprocessing."""
    global LOGGER
    LOGGER = utils.get_logger(args.verbose)

    dst_dir = args.dst_dir
    utils.check_dir(dst_dir)
    file_list_dir = args.file_list_dir
    utils.check_dir(file_list_dir)

    wordvec_path = os.path.join(args.wordvec)
    wordvec = utils.load_wordvec(wordvec_path)
    subword_embedding_path = os.path.join(args.embedding)
    subword_embedding = utils.load_wordvec(subword_embedding_path)

    subset_list = ['train.lat.txt', 'cv.lat.txt', 'test.lat.txt']
    processed_subset_list = []

    for subset in subset_list:
        subset_name = subset.split('.')[0] + '.' + subset.split('.')[2]
        preprocessed_list_file = os.path.join(args.processed_file_list_dir, subset_name)
        utils.remove_file(preprocessed_list_file)
        processed_subset_list.append(preprocessed_list_file)

    for i, subset in enumerate(subset_list):
        lat_file_list = os.path.join(file_list_dir, subset)

        # Compile the list of lat.gz files to process
        lattice_list = []
        with open(os.path.abspath(lat_file_list), 'r') as file_in:
            for line in file_in:
                lattice_list.append(line.strip())

        with Pool(args.num_threads) as pool:
            pool.starmap(process_one_lattice, zip(lattice_list, repeat(dst_dir),
                                                repeat(wordvec), repeat(subword_embedding),
                                                repeat(args.embed_apostrophe),
                                                repeat(args.uniform_subword_durations),
                                                repeat(processed_subset_list[i]))
            )

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
