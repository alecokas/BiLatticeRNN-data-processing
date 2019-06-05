"""Preprocess lattices."""
#!/usr/bin/env python3

import os
import sys
import gzip
import argparse
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import re
from toposort import toposort_flatten

import utils

EMBEDDING_LENGTH = 50
MAX_ARC_INFO = 10
SUBWORD_PROPERTIES = 2
HEADER_LINE_COUNT = 7


def read_lattice(lattice_path, subword_embedding):
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
    """
    utils.check_file(lattice_path)
    nodes = []
    edges = []
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
            # post = float(line[5].split('=')[1])
            grapheme_info = get_grapheme(line[5].split('=')[1], subword_embedding)
            edges.append([parent, child, am_score, lm_score] + grapheme_info)
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
    return nodes, edges, dependency, child_2_parent, parent_2_child

def get_grapheme(grapheme_info, subword_embedding):
    """ Separate all graphemes and durations into two separate lists.
        Return the result of concatenating these two lists.
    """
    token_durs = []
    token_list = []
    subword_list = grapheme_info.split(':')[1:-1]
    for subword_info in subword_list:
        subword, subword_dur = subword_info.split(',')
        token_durs.append(subword_dur)
        token = strip_phone(subword, 1, False)
        token_list.append(subword_embedding[token].tolist())
    return token_list + token_durs


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

def process_one_lattice(lattice_path, dst_dir, wordvec, subword_embedding):
    """Process a single lattice.

    Arguments:
        lattice_path {str} -- absolute path to lattices `.lat.gz`
        dst_dir {str} -- absolute path to destination directory
        wordvec {dict} -- word vector by calling `load_wordvec`
        subword_embedding {dict} -- subword embeddings
    """
    name = lattice_path.split('/')[-1].split('.')[0] + '.npz'
    try:
        LOGGER.info(name)
        name = os.path.join(dst_dir, name)
        if not os.path.isfile(name):
            nodes, edges, dependency, child_2_parent, parent_2_child = \
                read_lattice(lattice_path, subword_embedding)
            topo_order = toposort_flatten(dependency)
            # posterior = arc_posterior(lattice_path)
            # for each edge, the information contains
            # [EMBEDDING_LENGTH, duration(1), AM(1), LM(1), arc_posterior(1) or grapheme_info(MAX_ARC_INFO * 2)]
            edge_data = np.empty((len(edges), EMBEDDING_LENGTH + 1 + 1 + 1 + MAX_ARC_INFO * SUBWORD_PROPERTIES))
            mask_data = np.empty((len(edges), EMBEDDING_LENGTH + 1 + 1 + 1 + MAX_ARC_INFO * SUBWORD_PROPERTIES))
            ignore = []
            for i, edge in enumerate(edges):
                start_node = edge[0]
                end_node = edge[1]
                time = nodes[end_node][0] - nodes[start_node][0]
                word = nodes[end_node][1]

                padded_subword_data, subword_mask = pad_subword(edge[4])
                word_edge_data = np.concatenate(
                    (wordvec[word], np.array([time, edge[2], edge[3]])), axis=0)

                edge_data[i] = np.append(word_edge_data, padded_subword_data)

                ones = len(word_edge_data) * [1]
                mask_data[i] = np.append(ones, subword_mask)

                if word in ['<s>', '</s>', '!NULL', '<hes>']:
                    ignore.append(i)
            # save multiple variables into one .npz file
            np.savez(name, topo_order=topo_order, child_2_parent=child_2_parent,
                     parent_2_child=parent_2_child, edge_data=edge_data, mask_data=mask_data,
                     ignore=ignore)
    except OSError as exception:
        LOGGER.info('%s\n' %lattice_path + str(exception))

def pad_subword(subword_edge_features):
    """ The subword data (graphemic / phonetic) can be of variable length. In order to store
        this data in a numpy array, one assumes a maximum subword length and pad all shorter
        subwords to that length.
    """
    mask = []
    total_pad_count = MAX_ARC_INFO * SUBWORD_PROPERTIES - len(subword_edge_features)
    pads_per_chunk = int(total_pad_count / SUBWORD_PROPERTIES)
    subword_edge_data = list(chunks(subword_edge_features, SUBWORD_PROPERTIES))

    padded_subword_data = []
    for edge_chunk in subword_edge_data:
        padded_subword_data = padded_subword_data + edge_chunk + pads_per_chunk * [0]
        mask = mask + len(edge_chunk) * [1] + pads_per_chunk * [0]
    return padded_subword_data, mask

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
        help='The directory containing the files with the lists of lattice absolute paths for each subset'
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

    subset_list = ['train.txt', 'cv.txt', 'test.txt']

    for subset in subset_list:
        file_list = os.path.join(file_list_dir, subset)
        subword_embedding = load_wordvec(subword_embedding_path)
        wordvec = load_wordvec(wordvec_path)

        lattice_list = []
        with open(os.path.abspath(file_list), 'r') as file_in:
            for line in file_in:
                lattice_list.append(line.strip())

        with Pool(args.num_threads) as pool:
            pool.starmap(process_one_lattice, zip(lattice_list, repeat(dst_dir),
                                                repeat(wordvec), repeat(subword_embedding)))

if __name__ == '__main__':
    main()

