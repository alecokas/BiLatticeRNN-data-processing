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
POSN_INFO_LEN = 2
APOSTROPHE_TOKEN = 'A'

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
                    if len(line) == 5:
                        self.has_graphemes = True
                    if self.ignore_graphemes and self.has_graphemes:
                        line = line[:-1]
                        self.has_graphemes = False
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

    def convert_to_lattice(self, wordvec_dict, subword_embedding, dst_dir, log,dec_tree,
                           ignore_time_seg, processed_file_list_path=None, embed_apostrophe=False):
        """Convert confusion network object to lattice `.npz` format."""
        oov = set()
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
                    if edge_info[0] in wordvec_dict:
                        wordvec = wordvec_dict[edge_info[0]]
                    else:
                        oov.add(edge_info[0])
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
                    grapheme_feature_array = utils.get_grapheme_info(self.cn_arcs[i][4], subword_embedding, embed_apostrophe)
                    grapheme_data.append(grapheme_feature_array)

        npz_file_name = os.path.join(dst_dir, self.name + '.npz')
        if self.has_graphemes:
            # go through the array now and put it in a big masked array so it is just ine simple numpy array (I, J, F)
            max_grapheme_seq_length = utils.longest_grapheme_sequence(grapheme_data)
            padded_grapheme_data = np.empty((len(grapheme_data), max_grapheme_seq_length, LEN_GRAPHEME_FEATURES))
            mask = np.empty_like(padded_grapheme_data, dtype=bool)

            for arc_num, grapheme_seq in enumerate(grapheme_data):
                padded_grapheme_data[arc_num, :, :], mask[arc_num, :, :] = utils.pad_subword_sequence(grapheme_seq, max_grapheme_seq_length)

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
            utils.append_to_file(os.path.abspath(npz_file_name), processed_file_list_path)
        return oov

def process_one_cn(cn_path, dst_dir, wordvec_dict, subword_embedding, log, dec_tree,
                   ignore_time_seg, processed_file_list_path=None, embed_apostrophe=False):
    """Process a single confusion network.

    Arguments:
        cn_path {str} -- absolute path to a confusion network `.scf.gz`
        dst_dir {str} -- absolute path to destination directory
        wordvec_dict {dict} -- word vector by calling `load_wordvec`
        dec_tree {str} -- absolute path to decision tree object
    """
    name = cn_path.split('/')[-1].split('.')[0] + '.npz'
    LOGGER.info(name)
    confusion_net = CN(cn_path)
    oov = confusion_net.convert_to_lattice(
        wordvec_dict, subword_embedding, dst_dir, log, dec_tree,
        ignore_time_seg, processed_file_list_path, embed_apostrophe
    )
    return oov

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
        help='The directory containing the files with the lists of lattice absolute paths for each subset (*.cn.txt)'
    )
    parser.add_argument(
        '-p', '--processed-file-list-dir', type=str,
        help='The directory in which to save files with paths to the processed confusion networks (*.txt).'
    )
    parser.add_argument(
        '-l', '--log', default=False, action='store_true',
        help='Use posterior probabilities in log domain'
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
        '--decision-tree', type=str, dest='dec_tree', required=False, default='NONE'
    )
    parser.add_argument(
        '--embed-apostrophe', dest='embed_apostrophe', action='store_true'
    )
    parser.set_defaults(embed_apostrophe=False)
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
    wordvec = utils.load_wordvec(wordvec_path)
    subword_embedding_path = os.path.join(args.embedding)
    subword_embedding = utils.load_wordvec(subword_embedding_path)

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
        all_oov = set()
        for cn in cn_list:
            file_name = cn.split('/')[-1]
            print('Processing {}'.format(file_name[:-7]))
            oov = process_one_cn(
                cn, args.dst_dir, wordvec, subword_embedding, args.log,
                args.dec_tree, args.ignore_time_seg, processed_subset_list[i],
                args.embed_apostrophe
            )
            all_oov.update(oov)
    if not all_oov:
        print('OOV words were detected which could not be mapped to an embedding\nThese are the words:\n{}'.format(all_oov))
        with open('oov-words.txt', 'w') as oov_file:
            oov_file.write('\n'.join(list(all_oov)))


if __name__ == '__main__':
    main()
