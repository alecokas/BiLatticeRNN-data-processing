#!/usr/bin/env python3
"""Script for generating arc labels."""

import argparse
from cn_preprocess import CN
from itertools import repeat
from lattice_arc_label import tagging, load_baseline
import levenshtein_arc_label
from multiprocessing import Pool
import numpy as np
import os
import sys
import utils

def cn_pass_lev(cn_path, np_cn_path, start_frame, ctm_file, coeff=0.5):
    """Forward pass thourgh the confusion network.
    Return indices of arcs on the one-best path and corresponding sequence,
    and label of each arc.
    """
    confusion_net = CN(cn_path, ignore_graphemes=True)
    cum_sum = np.cumsum([0] + confusion_net.num_arcs)
    assert cum_sum[-1] == len(confusion_net.cn_arcs), "Wrong number of arcs."
    edge_labels = []
    sequence, indices = [], []

    if os.path.isfile(cn_path) and os.path.isfile(np_cn_path):
        tags = levenshtein_arc_label.levenshtein_tagging(ctm_file, cn_path, np_cn_path)
        for i in range(confusion_net.num_sets):
            tmp_post, tmp_edge, tmp_idx = -float('inf'), None, None
            for j in range(cum_sum[i], cum_sum[i+1]):
                edge_info = confusion_net.cn_arcs[j]
                edge_label = tags[j]
                edge_labels.append(edge_label)
                if edge_info[3] > tmp_post:
                    tmp_post = edge_info[3]
                    tmp_idx = j
                    tmp_edge = edge_info
            sequence.append(tmp_edge[0])
            indices.append(tmp_idx)

        clipped_indices = []
        clipped_seq = []
        ignore = ['!NULL', '<s>', '</s>', '<hes>']
        for i, j in zip(sequence, indices):
            if i not in ignore:
                clipped_seq.append(i)
                clipped_indices.append(j)
        return clipped_indices, clipped_seq, edge_labels
    else:
        raise Exception('File does not exist!')
        return [], [], []

def label(lattice_path, ctm_dir, dst_dir, baseline_dict, np_conf_dir, threshold=0.1):
    """Read HTK confusion networks and label each arc."""
    name = lattice_path.split('/')[-1].split('.')[0]
    np_lattice_path = os.path.join(np_conf_dir, name + '.npz')
    target_name = os.path.join(dst_dir, name + '.npz')

    if not os.path.isfile(target_name):
        name_parts = name.split('_')
        prefix = name_parts[0]
        ctm_file = os.path.join(ctm_dir, prefix + '.npz')
        try:
            start_frame = int(name_parts[-2])/100.
            indices, seq_1, edge_labels = cn_pass_lev(lattice_path, np_lattice_path, start_frame, ctm_file, threshold)
            if indices and seq_1:
                target = np.array(edge_labels, dtype='f')
                print("%s\t\t%f" %(name, np.mean(target)))
                ref, seq_2 = baseline_dict[name]
                assert len(indices) == len(ref)
                assert seq_1 == seq_2
                np.savez(target_name, target=target, indices=indices, ref=ref)
            else:
                print("Warning: Skiped target {} - Empty label sequence for the one-best".format(target_name))
                LOGGER.error("Warning: Skiped target {} - Empty label sequence for the one-best".format(target_name))
        except KeyError:
            LOGGER.error("ERROR: baseline does not contain this lattice %s" %name)
        except AssertionError:
            LOGGER.error("ERROR: reference and one-best do not match")
            LOGGER.error(name, indices, ref)
            LOGGER.error(seq_1, seq_2)

def main(args):
    """Main function for lattice arc tagging."""

    global LOGGER
    LOGGER = utils.get_logger(args.verbose, log_file_name=os.path.join(args.file_list_dir, 'targets-info'))

    dst_dir = os.path.join(args.dst_dir, 'target_overlap_{}'.format(args.threshold))
    utils.mkdir(dst_dir)

    np_conf_dir = os.path.join(args.processed_npz_dir)
    ctm_dir = os.path.join(args.ctm_dir)
    baseline_dict = load_baseline(args.one_best)

    file_list = []
    subset_list = ['train.cn.txt', 'cv.cn.txt', 'test.cn.txt']
    for subset in subset_list:
        file_list.append(os.path.join(args.file_list_dir, subset))

    cn_list = []
    for cn_subset_file in file_list:
        with open(os.path.abspath(cn_subset_file), 'r') as file_in:
            for path_to_lat in file_in:
                cn_list.append(path_to_lat.strip())

    for cn in cn_list:
        label(cn, ctm_dir, dst_dir, baseline_dict, np_conf_dir, args.threshold)

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='lattice pre-processing')
    parser.add_argument(
        '-d', '--dst-dir', type=str, required=True,
        help='The directory in which to save the output files'
    )
    parser.add_argument(
        '-p', '--processed-npz-dir', type=str, required=True,
        help='The directory containing the processed confusion network / lattice numpy files'
    )
    parser.add_argument(
        '-f', '--file-list-dir', type=str,
        help='The directory containing the file lists for the train, cv, and test sets.'
    )
    parser.add_argument(
        '-b', '--one-best', type=str,
        required=True,
        help='Path to the one best transcription (*.mlf.det file) for the confusion network files given in file-list-dir.'
    )
    parser.add_argument(
        '-ctm', '--ctm-dir', type=str, required=True,
        help='Directory containing reference CTM files for arc tagging (*.stm file).'
    )
    parser.add_argument(
        '-n', '--num-threads',
        help='number of threads to use for concurrency',
        type=int, default=30
    )
    parser.add_argument(
        '-t', '--threshold',
        help='Cut-off threshold for tagging decision',
        type=float, default=0.1
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Set logging level: ERROR (default), '\
             'WARNING (-v), INFO (-vv), DEBUG (-vvv)',
        action='count', default=0
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
