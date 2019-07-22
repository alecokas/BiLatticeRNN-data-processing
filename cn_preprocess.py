"""Preprocess confusion networks into `.npz` lattice format."""

import argparse
import gzip
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import numpy.ma as ma
import os
import pickle
import re
from trees import Tree
import utils


LEN_GRAPHEME_FEATURES = 5


class CN:
    """Confusion networks from file."""

    def __init__(self, path, ignore_graphemes=False):
        """Confusion network constructor."""
        self.path = path
        self.cn_arcs = []
        self.num_sets = None
        self.num_arcs = []
        self.has_graphemes = False
        self.ignore_graphemes = ignore_graphemes

        utils.check_file(self.path)
        self.name = self.path.split('/')[-1].strip('.scf.gz')
        self.load()

    def load(self):
        """Load `.scf.gz` file into CN object."""
        with gzip.open(self.path, 'rt', encoding='utf-8') as file_in:
            line = file_in.readline().split('=')
            assert line[0] == 'N', "Problem with the first line, expected N=num_sets."
            self.num_sets = int(line[1])
            for i in range(self.num_sets):
                line = file_in.readline().split('=')
                assert line[0] == 'k', "Problem with the start of the set, expected k=num_arcs."
                num_arcs = int(line[1])
                self.num_arcs.append(num_arcs)
                for _ in range(num_arcs):
                    line = file_in.readline().split()
                    # W=word s=start e=end p=posterior (Optionally d=grapheme-information)
                    if self.ignore_graphemes:
                        line = line[:-1]
                    if len(line) == 5:
                        self.has_graphemes = True
                    if len(line) == 4 or len(line) == 5:
                         self.cn_arcs.append([item.split('=')[1] for item in line])
                    else:
                        raise Exception('Unexpected format - all arc lines should contain 4 or 5 elements')
        # Convert all numbers to float, and p in log domain
        for i, arc in enumerate(self.cn_arcs):
            self.cn_arcs[i][1:4] = [float(arc[i]) for i in range(1, 4)]
        # Reverse lists, as confusion sets in `.scf.gz` file are reversed
        self.cn_arcs.reverse()
        self.num_arcs.reverse()

    def convert_to_lattice(self, wordvec_dict, subword_embedding, dst_dir, log, dec_tree, ignore_time_seg, processed_file_list_path=None):
        """Convert confusion network object to lattice `.npz` format."""
        utils.mkdir(dst_dir)
        if ignore_time_seg != False:
            ignore_time_seg_dict = np.load(ignore_time_seg)
            ignore_time_seg_dict = ignore_time_seg_dict['ignore'][()]
        else:
            ignore_time_seg_dict = False
        topo_order = list(range(self.num_sets + 1))
        cum_sum = np.cumsum([0] + self.num_arcs)
        assert cum_sum[-1] == len(self.cn_arcs), "Wrong number of arcs."
        edge_data = []
        start_times = []
        grapheme_data = []
        child_2_parent = {}
        parent_2_child = {}
        ignore = []
        use_dec_tree=False
        if dec_tree != 'NONE':
            with open(dec_tree, 'rb') as dec_tree:
                decision_tree=pickle.load(dec_tree)
            use_dec_tree=True
        else:
            use_dec_tree=False
        for i in range(self.num_sets):
            parent_2_child[i] = {i+1: list(range(cum_sum[i], cum_sum[i+1]))}
            child_2_parent[i+1] = {i: list(range(cum_sum[i], cum_sum[i+1]))}
            for j in range(cum_sum[i], cum_sum[i + 1]):
                edge_info = self.cn_arcs[j]
                if edge_info[0] == '!NULL':
                    wordvec = np.zeros_like(wordvec_dict['<hes>'])
                else:
                    wordvec = wordvec_dict[edge_info[0]]
                if log:
                    if use_dec_tree:
                        conf=np.exp(edge_info[3])
                        conf=decision_tree.conv_value(conf)
                        conf=np.log(conf)
                    else:
                        conf=edge_info[3]
                    edge_vec = np.concatenate(
                        (wordvec, np.array([edge_info[2] - edge_info[1],
                                            conf])), axis=0)
                else:
                    if use_dec_tree:
                        conf=np.exp(edge_info[3])
                        conf=decision_tree.conv_value(conf)
                    else:
                        conf=np.exp(edge_info[3])
                    edge_vec = np.concatenate(
                        (wordvec, np.array([edge_info[2] - edge_info[1],
                                            conf])), axis=0)
                start_times.append(edge_info[1])
                edge_data.append(edge_vec)
                if edge_info[0] in ['<s>', '</s>', '!NULL', '<hes>']:
                    ignore.append(j)
                elif ignore_time_seg != False:
                    file_name = self.name
                    name = file_name.split("_")[0]
                    start_frame = file_name.split("_")[-2]
                    start_frame = float(start_frame)/100
                    start_time = edge_info[1] + start_frame
                    end_time = edge_info[2] + start_frame
                    if name in ignore_time_seg_dict:
                        for tup in ignore_time_seg_dict[name]:
                            if start_time > tup[0] and end_time < tup[1]:
                                ignore.append(j)
                
                # Deal with any grapheme data if required:
                if self.has_graphemes:
                    grapheme_feature_array = get_grapheme_info(self.cn_arcs[i][4], subword_embedding)
                    grapheme_data.append(grapheme_feature_array)

        npz_file_name = os.path.join(dst_dir, self.name + '.npz')
        if self.has_graphemes:
            # go through the array now and put it in a big masked array so it is just ine simple numpy array (I, J, F)
            max_grapheme_seq_length = longest_grapheme_sequence(grapheme_data)
            padded_grapheme_data = np.empty((len(grapheme_data), max_grapheme_seq_length, LEN_GRAPHEME_FEATURES))
            mask = np.empty_like(padded_grapheme_data, dtype=bool)

            for arc_num, grapheme_seq in enumerate(grapheme_data):
                padded_grapheme_data[arc_num, :, :], mask[arc_num, :, :] = pad_subword_sequence(grapheme_seq, max_grapheme_seq_length)

            masked_grapheme_data = ma.masked_array(padded_grapheme_data, mask=mask, fill_value=-999999)
            np.savez(npz_file_name,
                    topo_order=topo_order, child_2_parent=child_2_parent,
                    parent_2_child=parent_2_child, edge_data=np.asarray(edge_data),
                    ignore=ignore, grapheme_data=masked_grapheme_data, start_times=start_times)
        else:
            np.savez(npz_file_name,
                    topo_order=topo_order, child_2_parent=child_2_parent,
                    parent_2_child=parent_2_child, edge_data=np.asarray(edge_data),
                    ignore=ignore, start_times=start_times)

        if processed_file_list_path is not None:
            append_path_to_txt(os.path.abspath(npz_file_name), processed_file_list_path)

def append_path_to_txt(path_to_add, target_file):
    with open(target_file, "a") as file:
        file.write(path_to_add + '\n')

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

def longest_grapheme_sequence(grapheme_list):
    max_length_seq = -1
    for arc in grapheme_list:
        seq_length = arc.shape[0]
        if seq_length > max_length_seq:
            max_length_seq = seq_length
    if max_length_seq == -1:
        raise Exception('max_length never updated')
    # print('Max length: {}'.format(max_length_seq))
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
    mask = np.concatenate((valid_array, invalid_array), axis=0)
    return padded_subword_seq, mask

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

def load_wordvec(path):
    """Load pre-computed word vectors.

    Arguments:
        path {string} -- path to `.npy` file contains a dictionary of all words
            and their word vectors

    Returns:
        dictionary -- word vector
    """
    utils.check_file_logging(LOGGER, path)
    wordvec = np.load(path).item()
    return wordvec

def process_one_cn(cn_path, dst_dir, wordvec_dict, subword_embedding, log, dec_tree, ignore_time_seg, processed_file_list_path=None):
    """Process a single confusion network.

    Arguments:
        cn_path {str} -- absolute path to a confusion network `.scf.gz`
        dst_dir {str} -- absolute path to destination directory
        wordvec_dict {dict} -- word vector by calling `load_wordvec`
        dec_tree {str} -- absolute path to decision tree object
    """
    name = cn_path.split('/')[-1].split('.')[0] + '.npz'
    try:
        LOGGER.info(name)
        confusion_net = CN(cn_path)
        confusion_net.convert_to_lattice(wordvec_dict, subword_embedding, dst_dir, log, dec_tree, ignore_time_seg, processed_file_list_path)
    except OSError as exception:
        print('OSError: {}'.format(cn_path))
        LOGGER.info('%s\n' %cn_path + str(exception))

def main():
    """Main function for converting CN into `.npz` lattices."""
    parser = argparse.ArgumentParser(
        description='confusion network pre-processing'
    )
    parser.add_argument(
        '-d', '--dst-dir', type=str,
        help='Location to save the processed confusion network files (*.npz)'
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
        help='The directory in which to save files with paths to the processed confusion networks (*.txt).'
    )  
    parser.add_argument(
        '-l', '--log', default=False, action='store_true',
        help='use posterior probabilities in log domain'
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Set logging level: ERROR (default), '\
             'WARNING (-v), INFO (-vv), DEBUG (-vvv)',
        action='count', default=0
    )
    parser.add_argument(
        '-n', '--num_threads',
        help='number of threads to use for concurrency',
        type=int, default=30
    )
    parser.add_argument(
        '--decision_tree', type=str, dest='dec_tree', required=False, default='NONE'
    )
    parser.add_argument(
        '--ignore_time_seg', dest='ignore_time_seg', required=False, default=False
    )
    args = parser.parse_args()

    global LOGGER
    LOGGER = utils.get_logger(args.verbose)

    dst_dir = args.dst_dir
    utils.check_dir(dst_dir)
    file_list_dir = args.file_list_dir
    utils.check_dir(file_list_dir)

    wordvec_path = os.path.join(args.wordvec)
    wordvec = load_wordvec(wordvec_path)
    subword_embedding_path = os.path.join(args.embedding)
    subword_embedding = load_wordvec(subword_embedding_path)

    subset_list = ['train.cn.txt', 'cv.cn.txt', 'test.cn.txt']
    processed_subset_list = []

    for subset in subset_list:
        subset_name = subset.split('.')[0] + '.' + subset.split('.')[2]
        preprocessed_list_file = os.path.join(args.processed_file_list_dir, subset_name)
        utils.remove_file(preprocessed_list_file)
        processed_subset_list.append(preprocessed_list_file)

    for i, subset in enumerate(subset_list):
        lat_file_list = os.path.join(file_list_dir, subset)

        # Compile the list of CN files to process
        cn_list = []
        with open(os.path.abspath(lat_file_list), 'r') as file_in:
            for line in file_in:
                cn_list.append(line.strip())

        for cn in cn_list:
            file_name = cn.split('/')[-1]
            print('Processing {}'.format(file_name[:-7]))
            process_one_cn(
                cn, args.dst_dir, wordvec, subword_embedding, args.log,
                args.dec_tree, args.ignore_time_seg, processed_subset_list[i]
            )


if __name__ == '__main__':
    main()
