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
import utils

def cn_pass_lev(cn_path, np_cn_path, start_frame, stm_file, coeff=0.5):
    """Forward pass thourgh the confusion network.
    Return indices of arcs on the one-best path and corresponding sequence,
    and label of each arc.
    """
    confusion_net = CN(cn_path, ignore_graphemes=True)
    cum_sum = np.cumsum([0] + confusion_net.num_arcs)
    assert cum_sum[-1] == len(confusion_net.cn_arcs), "Wrong number of arcs."
    edge_labels = []
    sequence, indices = [], []
    tags = levenshtein_arc_label.levenshtein_tagging(stm_file, cn_path, np_cn_path)
    for i in range(confusion_net.num_sets):
        tmp_post, tmp_edge, tmp_idx = -float('inf'), None, None
        for j in range(cum_sum[i], cum_sum[i+1]):
            edge_info = confusion_net.cn_arcs[j]
            # start = start_frame + edge_info[1]
            # end = start_frame + edge_info[2]
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

#unedited original script
def cn_pass(cn_path, start_frame, stm, coeff=0.5):
    """Forward pass thourgh the confusion network.
    Return indices of arcs on the one-best path and corresponding sequence,
    and label of each arc.
    """
    confusion_net = CN(cn_path)
    cum_sum = np.cumsum([0] + confusion_net.num_arcs)
    assert cum_sum[-1] == len(confusion_net.cn_arcs), "Wrong number of arcs."
    edge_labels = []
    sequence, indices = [], []
    for i in range(confusion_net.num_sets):
        tmp_post, tmp_edge, tmp_idx = -float('inf'), None, None
        for j in range(cum_sum[i], cum_sum[i+1]):
            edge_info = confusion_net.cn_arcs[j]
            start = start_frame + edge_info[1]
            end = start_frame + edge_info[2]
            edge_label = tagging(stm, (start, end), edge_info[0], coeff)
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

def label(lattice_path, stm_dir, dst_dir, baseline_dict, np_conf_dir, lev, threshold=0.5):
    """Read HTK confusion networks and label each arc."""
    name = lattice_path.split('/')[-1].split('.')[0]
    np_lattice_path = os.path.join(np_conf_dir, name + '.npz')
    target_name = os.path.join(dst_dir, name + '.npz')

    if not os.path.isfile(target_name):
        name_parts = name.split('_')
        prefix = name_parts[0]
        stm_file = os.path.join(stm_dir, prefix + '.npz')
        try:
            # stm = np.load(stm_file)
            start_frame = int(name_parts[-2])/100.
            
            if lev:
                indices, seq_1, edge_labels = cn_pass_lev(lattice_path, np_lattice_path, start_frame, stm_file, threshold)
            else:
                # TODO: Don't think this is correct, so I have commented it out
                # indices, seq_1, edge_labels = cn_pass(lattice_path, start_frame, stm, threshold)
                raise NotImplementedError('cn_pass() does not return anything and so I do not trust this function')
            target = np.array(edge_labels, dtype='f')
            print("%s\t\t%f" %(name, np.mean(target)))
            ref, seq_2 = baseline_dict[name]
            assert len(indices) == len(ref)
            assert seq_1 == seq_2
            np.savez(target_name, target=target, indices=indices, ref=ref)
        except IOError:
            print("ERROR: file does not exist: %s" %stm_file)
        except KeyError:
            print("ERROR: baseline does not contain this lattice %s" %name)
        except AssertionError:
            print("ERROR: reference and one-best do not match")
            print(name, indices, ref)
            print(seq_1, seq_2)

def main():
    """Main function for lattice arc tagging."""
    parser = argparse.ArgumentParser(description='lattice pre-processing')
    parser.add_argument(
        '-stm', '--stm-dir', type=str, required=True,
        help='Directory containing reference stm files for arc tagging (*.stm file).'
    )
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
        '-n', '--num-threads',
        help='number of threads to use for concurrency',
        type=int, default=30
    )
    parser.add_argument(
        '-t', '--threshold',
        help='Cut-off threshold for tagging decision',
        type=float, default=0.5
    )
    parser.add_argument(
        '--lev', dest='lev', action='store_true', default=False, help='Use Levenshtein distance metric for arc tagging'
    )
    args = parser.parse_args()

    dst_dir = os.path.join(args.dst_dir, 'target_overlap_{}'.format(args.threshold))
    utils.mkdir(dst_dir)

    np_conf_dir = os.path.join(args.processed_npz_dir)
    stm_dir = os.path.join(args.stm_dir)
    baseline_dict = load_baseline(args.one_best)

    subset_list = ['train.txt', 'cv.txt', 'test.txt']
    for subset in subset_list:
        file_list = os.path.join(args.file_list_dir, subset)

    cn_list = []
    with open(os.path.abspath(file_list), 'r') as file_in:
        for path_to_lat in file_in:
            cn_list.append(path_to_lat.strip())

    for cn in cn_list:
        label(cn, stm_dir, dst_dir, baseline_dict, np_conf_dir, args.lev, args.threshold)

if __name__ == '__main__':
    main()
