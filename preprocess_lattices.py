""" Preprocess lattices from the *.lat.gz format into the processed *.npz format.
    At the same time, produce the file list with paths to the processed files.
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
# Grapheme embedding (4), grapheme duration (1)
LEN_GRAPHEME_FEATURES = 5


def read_lattice(lattice_path, subword_embedding=None):
    """Read HTK lattices.

    Arguments:
        lattice_path {string} -- absolute path of a compressed lattice
            in `.lat.gz` format
        subword_embedding -- subword embeddings

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

            # Set grapheme_dict to None so that if there is no grapheme information, this can
            # be used downstream as a check
            # grapheme_dict = None
            if line[5].split('=')[0] == 'r':
                # Remove the prnounciation information if it is present
                del line[5]
            if line[5].split('=')[0] == 'd':
                # Extract a grapheme feature vector of dimensions: (num_graphemes, num_features)
                grapheme_feature_array = get_grapheme_info(line[5].split('=')[1], subword_embedding)
                post_idx = 6
            else:
                post_idx = 5
            if line[post_idx].split('=')[0] == 'p':
                # Expect posterior information
                post = float(line[post_idx].split('=')[1])
            else:
                raise Exception('This lattice ({}) has an unrecognised arc parameter sequence'.format(lattice_path))

            edges.append([parent, child, am_score, lm_score, post])
            grapheme_data.append(grapheme_feature_array)

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
    max_grapheme_seq_length = longest_grapheme_sequence(grapheme_data)
    padded_grapheme_data = np.empty((len(grapheme_data), max_grapheme_seq_length, LEN_GRAPHEME_FEATURES))
    mask = np.empty_like(padded_grapheme_data, dtype=bool)

    for arc_num, grapheme_seq in enumerate(grapheme_data):
        padded_grapheme_data[arc_num, :, :], mask[arc_num, :, :] = pad_subword_sequence(grapheme_seq, max_grapheme_seq_length)

    masked_grapheme_data = ma.masked_array(padded_grapheme_data, mask=mask, fill_value=-999999)

    return nodes, edges, dependency, child_2_parent, parent_2_child, masked_grapheme_data

def get_grapheme_info(grapheme_info, subword_embedding):
    """ Extract grapheme information and store it in an array with the following form:
        ((emb-0-0, emb-0-1, emb-0-2, emb-0-3, dur-0)
            .       .         .        .       .
            .       .         .        .       .
            .       .         .        .       .
        (emb-J-0, emb-J-1, emb-J-2, emb-J-3, dur-J))
    """
    subword_list = grapheme_info.split(':')[1:-1]
    grapheme_feature_list = np.empty((len(subword_list), LEN_GRAPHEME_FEATURES))
    for i, subword_info in enumerate(subword_list):
        subword, subword_dur = subword_info.split(',')[:2]
        token = strip_phone(subword, 1, False)
        if subword_embedding is None:
            raise Exception('No subword embedding!')
        else:
            grapheme_feature_list[i, :] = np.append(subword_embedding[token], subword_dur)
    return grapheme_feature_list


def strip_phone(phone_info, phone_context_width, incl_posn_info):
    """ Strip phones of context and optionally the location indicator

        Arguments:
            phone_info: String with the full phone context information and location indicators.
            phone_context_width: The phone context width as an integer
            incl_posn_info: A boolean indicator for whether or not to include the phone position information (^I, ^M, ^F)
    """
    if phone_context_width > 3:
        raise Exception('The phone context width cannot be greater than 3.')

    itemised_phone_info = re.split(r'\+|\-', phone_info)
    if len(itemised_phone_info) == 1:
        return itemised_phone_info[0] if incl_posn_info else remove_location_indicator(itemised_phone_info[0])
    elif len(itemised_phone_info) == 3:
        if phone_context_width > 1:
            # Assume that if the context is 2 (bigram), we want the include the preceding phone
            stop = phone_context_width
            return itemised_phone_info[:stop] if incl_posn_info else remove_location_indicator(itemised_phone_info[:stop])
        else:
            return itemised_phone_info[1] if incl_posn_info else remove_location_indicator(itemised_phone_info[1])
    else:
        raise Exception('The phone length should be 1 or 3, but found {}'.format(len(itemised_phone_info)))

def remove_location_indicator(phone_with_location):
    """ Strip location indicators from a string or strings within a list and return the result as a string

        Arguments:
            phone_with_location: Either a string or list containing the raw phone with location indicators.
    """
    if isinstance(phone_with_location, list):
        clean_phone_list = []
        for phone in phone_with_location:
            clean_phone_list.append(phone.split('^')[0])
        return ' '.join(clean_phone_list)
    else:
        return phone_with_location.split('^')[0]

def longest_grapheme_sequence(grapheme_list):
    max_length_seq = -1
    for arc in grapheme_list:
        seq_length = arc.shape[0]
        if seq_length > max_length_seq:
            max_length_seq = seq_length
    if max_length_seq == -1:
        raise Exception('max_length never updated')
    print('Max length: {}'.format(max_length_seq))
    return max_length_seq


def pad_subword_sequence(subword_seq, max_seq_length):
    """ The subword sequence (graphemic / phonetic) can be of variable length. In order to store
        this data in a numpy array, one pads and masks the subword dimension to the max sequence
        length.

        subword_seq: numpy array with dimensions (graphemes, features)
        max_seq_length: The length of the maximum subword sequence
    """
    pad_count = max_seq_length - subword_seq.shape[0]
    zero_pads = np.zeros((pad_count, LEN_GRAPHEME_FEATURES))
    padded_subword_seq = np.concatenate((subword_seq, zero_pads), axis=0)

    valid_array = np.ones_like(zero_pads, dtype=bool)
    invalid_array = np.zeros_like(subword_seq, dtype=bool)
    mask = np.concatenate((valid_array, invalid_array), axs=0)
    return padded_subword_seq, mask
    # return ma.masked_array(padded_subword_seq, mask=mask, fill_value=-999999)


def chunks(l, n):
    """Yield successive n-sized chunks from list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_wordvec(path):
    """Load pre-computed word vectors.

    Arguments:
        path {str} -- path to `.npy` file contains a dictionary of all words
            and their word vectors

    Returns:
        dictionary -- word vector
    """
    utils.check_file_logging(LOGGER, path)
    wordvec = np.load(path).item()
    return wordvec

def process_one_lattice(lattice_path, dst_dir, wordvec, subword_embedding,
                        processed_file_list_path=None):
    """Process a single lattice.

    Arguments:
        lattice_path {str} -- absolute path to lattices `.lat.gz`
        dst_dir {str} -- absolute path to destination directory
        wordvec {dict} -- word vector by calling `load_wordvec`
        subword_embedding {dict} -- subword embeddings
    """
    name = lattice_path.split('/')[-1].split('.')[0] + '.npz'
    print('Processing {}'.format(name))
    try:
        LOGGER.info(name)
        name = os.path.join(dst_dir, name)
        if not os.path.isfile(name):
            nodes, edges, dependency, child_2_parent, parent_2_child, grapheme_data \
                = read_lattice(lattice_path, subword_embedding)
            topo_order = toposort_flatten(dependency)

            # for each edge, the information contains
            # [EMBEDDING_LENGTH, duration(1), AM(1), LM(1), arc_posterior(1)]
            edge_data = np.empty((len(edges), EMBEDDING_LENGTH + 1 + 1 + 1 + 1))
            ignore = []
            for i, edge in enumerate(edges):
                start_node = edge[0]
                end_node = edge[1]
                time = nodes[end_node][0] - nodes[start_node][0]
                word = nodes[end_node][1]

                edge_data[i] = np.concatenate(
                    (wordvec[word], np.array([time, edge[2], edge[3], edge[4]])), axis=0)

                if word in ['<s>', '</s>', '!NULL', '<hes>']:
                    ignore.append(i)

            # save multiple variables into one .npz file
            np.savez(name, topo_order=topo_order, child_2_parent=child_2_parent,
                     parent_2_child=parent_2_child, edge_data=edge_data,
                     ignore=ignore, grapheme_data=grapheme_data
            )
            if processed_file_list_path is not None:
                append_path_to_txt(os.path.abspath(name), processed_file_list_path)
    except OSError as exception:
        LOGGER.info('%s\n' %lattice_path + str(exception))


def append_path_to_txt(path_to_add, target_file):
    with open(target_file, "a") as file:
        file.write(path_to_add + '\n')


def main():
    """Main function for lattice preprocessing."""
    parser = argparse.ArgumentParser(description='lattice pre-processing')

    parser.add_argument(
        '-d', '--dst-dir', type=str,
        help='Location to save the uncompressed lattice files (*.npz)'
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

    global LOGGER
    LOGGER = utils.get_logger(args.verbose)

    dst_dir = args.dst_dir
    utils.check_dir(args.dst_dir)
    file_list_dir = args.file_list_dir
    utils.check_dir(args.dst_dir)

    subword_embedding_path = os.path.join(args.embedding)
    wordvec_path = os.path.join(args.wordvec)

    subset_list = ['train.lat.txt', 'cv.lat.txt', 'test.lat.txt']
    processed_subset_list = []

    for subset in subset_list:
        subset_name = subset.split('.')[0] + '.' + subset.split('.')[2]
        preprocessed_list_file = os.path.join(args.processed_file_list_dir, subset_name)

        try:
            os.remove(preprocessed_list_file)
        except OSError:
            pass

        processed_subset_list.append(preprocessed_list_file)

    for i, subset in enumerate(subset_list):
        file_list = os.path.join(file_list_dir, subset)
        subword_embedding = load_wordvec(subword_embedding_path)
        wordvec = load_wordvec(wordvec_path)

        lattice_list = []
        with open(os.path.abspath(file_list), 'r') as file_in:
            for line in file_in:
                lattice_list.append(line.strip())

        with Pool(args.num_threads) as pool:
            pool.starmap(process_one_lattice, zip(lattice_list, repeat(dst_dir),
                                                repeat(wordvec), repeat(subword_embedding),
                                                repeat(processed_subset_list[i])))

if __name__ == '__main__':
    main()

