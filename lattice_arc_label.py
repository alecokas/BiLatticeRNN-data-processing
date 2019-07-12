#!/usr/bin/env python3
"""Script for generating arc labels."""

import os
import argparse
from bisect import bisect_left
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from preprocess_lattices import read_lattice
from toposort import toposort_flatten
import utils
# from preprocess import levenshtein

def score(array, grammar_scale=20.0):
    """Get the combined score from the vector."""
    # assert len(array) == 4, "lattice feature dimension wrong: it is {}, but should be 4".format(len(array))
    am_score = array[2]
    lm_score = array[3]
    return am_score / grammar_scale + lm_score

def lattice_pass(nodes, edges, dependency, child_dict, parent_dict):
    """Forward pass through the lattice to find the one-best path. (Viterbi)
    Return indices of the arcs in the lattice without start/end sentences.
    """
    topo_order = toposort_flatten(dependency)
    max_dict = {}
    node_state = [None] * len(nodes)
    edge_state = [None] * len(edges)
    node_state[0] = 0
    for each_node in topo_order:
        if each_node in child_dict:
            in_nodes, in_edges = zip(*child_dict[each_node].items())
            in_states = [edge_state[i] for i in in_edges]
            node_state[each_node] = np.amax(in_states)
            max_dict[each_node] = in_nodes[np.argmax(in_states)]

        if each_node in parent_dict:
            for each_edge in parent_dict[each_node].values():
                edge_state[each_edge] = node_state[each_node] \
                                        + score(edges[each_edge])

    start_node = topo_order[0]
    end_node = topo_order[-1]
    node_indices = [end_node]
    while node_indices[0] != start_node:
        node_indices.insert(0, max_dict[node_indices[0]])

    sequence = []
    for i in node_indices[1:]:
        sequence.append(nodes[i][1])
    indices = []
    for i in range(len(node_indices) - 1):
        indices.append(parent_dict[node_indices[i]][node_indices[i+1]])
    assert len(sequence) == len(indices), "inconsistent lenghts."

    clipped_indices = []
    clipped_seq = []
    ignore = ['!NULL', '<s>', '</s>', '<hes>']
    for i, j in zip(sequence, indices):
        if i not in ignore:
            clipped_seq.append(i)
            clipped_indices.append(j)
    return clipped_indices, clipped_seq

def load_baseline(file_name):
    """Load `mlf.det` files."""
    baseline_dict = {}
    result_mapping = {"CORRECT": 1.0, "SUBSTITUTION": 0.0, "INSERTION": 0.0,
                      "UNKNOWN": None}
    with open(file_name, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            if line.startswith("\""):
                utt_id = line.strip().strip("\"").split("/")[-1].strip(".lab")
                sequence = []
                ref_label = []
                while True:
                    next_line = next(file_in)
                    if next_line.startswith("."):
                        break
                    else:
                        word = next_line.split()[2]
                        result = next_line.split()[4]
                        sequence.append(word)
                        ref_label.append(result_mapping[result])
                baseline_dict[utt_id] = (ref_label, sequence)
    return baseline_dict

def tagging(stm, time_span, word, coeff=0.5):
    """Arc tagging policy based on overlapping time."""
    time = stm['time']
    time = np.append(time, [stm['time'][-1] + stm['duration'][-1]])
    start, end = time_span[0], time_span[1]
    left_pos = bisect_left(time, start)
    right_pos = bisect_left(time, end)

    # For the case of dummy node
    if (end - start) == 0 and word == '!NULL':
        return 1
    elif left_pos == len(time) or right_pos == 0:
        return 0
    else:
        tag = 0
        left_idx = max(0, left_pos - 1)
        right_idx = min(len(time) - 1, right_pos)
        for i in range(left_idx, right_idx):
            left_time = time[i]
            right_time = time[i+1]
            overlap = (min(right_time, end) - max(left_time, start)) \
                      / (max(right_time, end) - min(left_time, start))
            if overlap > coeff and word == stm['word'][i]:
                tag = 1
        return tag

def label(lattice_path, stm_dir, dst_dir, baseline_dict, threshold=0.5):
    """Read HTK lattice and label each arc."""
    name = lattice_path.split('/')[-1].split('.')[0] #BPL404-97376-20141126-024552-ou_COXXXXX_0024915_0025043
    target_name = os.path.join(dst_dir, name + '.npz')

    if not os.path.isfile(target_name):
        name_parts = name.split('_')
        prefix = name_parts[0]
        stm_file = os.path.join(stm_dir, prefix + '.npz') #taking npz file from data/*/stm/*.npz 
        try:
            stm = np.load(stm_file)
            start_frame = int(name_parts[-2])/100.

            nodes, edges, dependency, child_dict, parent_dict = read_lattice(lattice_path)
            for node in nodes:
                node[0] += start_frame
            edge_labels = []
            for edge in edges:
                time_span = (nodes[edge[0]][0], nodes[edge[1]][0])
                word = nodes[edge[1]][1]
                edge_label = tagging(stm, time_span, word, threshold)
                edge_labels.append(edge_label)
            target = np.array(edge_labels, dtype='f')
            print("%s\t\t%f" %(name, np.mean(target)))

            indices, seq_1 = lattice_pass(nodes, edges, dependency, child_dict,
                                          parent_dict)
            ref, seq_2 = baseline_dict[name]
            assert len(ref) == len(indices)
            # assert seq_1 == seq_2
            np.savez(target_name, target=target, indices=indices, ref=ref)
        except IOError:
            raise Exception("ERROR: file does not exist: %s" %stm_file)
        except KeyError:
            raise Exception("ERROR: baseline does not contain this lattice %s" %name)
        except AssertionError:
            print(name, indices, ref)
            print(seq_1, seq_2)
            raise Exception("ERROR: reference and one-best do not match")

    else:
        print('Warning: {} already exists - skipping'.format(target_name))

def main():
    """Main function for lattice arc tagging."""
    parser = argparse.ArgumentParser(description='lattice pre-processing')
    parser.add_argument(
        '-f', '--file-list-dir', type=str,
        help='The directory containing the file lists for the train, cv, and test sets.'
    )
    parser.add_argument(
        '-n', '--num-threads',
        help='number of threads to use for concurrency',
        type=int, default=30
    )
    parser.add_argument(
        '-t', '--threshold',
        help='cut-off threshold for tagging decision',
        type=float, default=0.5
    )
    parser.add_argument(
        '-b', '--one-best', type=str,
        default='/home/dawna/babel/BABEL_OP3_404/releaseB/exp-graphemic-ar527-v3/J2/decode-ibmseg-fcomb/scoring/sclite/dev/decode_rescore_tg_20.0_0.0_rescore_wlat_20.0_0.0_rescore_plat_20.0_0.0.mlf.det',
        help='Path to the one best transcription (*.mlf.det file) for the lattice files given in file-list-dir.'
    )
    parser.add_argument(
        '-stm', '--stm-dir', type=str, required=True,
        help='Directory containing reference stm files for arc tagging (*.stm file).'
    )
    parser.add_argument(
        '-d', '--dst-dir', type=str, required=True,
        help='The directory in which to save the output files'
    )
    args = parser.parse_args()
    
    dst_dir = os.path.join(args.dst_dir, 'target_overlap_{}'.format(args.threshold))
    utils.mkdir(dst_dir)
    baseline_dict = load_baseline(args.one_best)

    subset_list = ['train.txt', 'cv.txt', 'test.txt']
    for subset in subset_list:
        file_list = os.path.join(args.file_list_dir, subset)

    lattice_list = []
    with open(os.path.abspath(file_list), 'r') as file_in:
        for path_to_lat in file_in:
            lattice_list.append(path_to_lat.strip())

    with Pool(args.num_threads) as pool:
        pool.starmap(label, zip(lattice_list, repeat(args.stm_dir), repeat(dst_dir),
                                repeat(baseline_dict), repeat(args.threshold)))

if __name__ == '__main__':
    main()
