import argparse
import math
import numpy as np
import os
import re
import sys
import utils


WORD_EMBEDDING_LENGTH = 50
DURATION_IDX = 50
AM_INDEX = 51
LM_INDEX = 52
POST_IDX = 53
FRAME_PERIOD = 0.01


def set_of_processed_file_names(directory_to_search, remove_extension=False, extension='.npz'):
    processed_names = set()
    for _, _, names in os.walk(directory_to_search):
        for name in names:
            if name.endswith(extension):
                if remove_extension:
                    name = name[:-len(extension)]
                processed_names.add(name)
    return processed_names

def num_added_features(include_am, include_lm):
    added_feature_count = 0
    if include_am:
        added_feature_count += 1
    if include_lm:
        added_feature_count += 1
    return added_feature_count

def cost_fn(x1_start_time, x1_dur, x2_start_time, x2_dur, max_time_diff):
    x1_end_time = x1_start_time + x1_dur
    x2_end_time = x2_start_time + x2_dur

    start_time_diff = abs(x1_start_time - x2_start_time)
    end_time_diff = abs(x1_end_time - x2_end_time)
    dur_diff = abs(x1_dur - x2_dur)

    if start_time_diff > max_time_diff and end_time_diff > max_time_diff:
        # disquality all arcs where the start and end time difference is greater than a max_time_diff
        return -1

    return math.sqrt((start_time_diff ** 2) + (end_time_diff ** 2) + (dur_diff ** 2))

def find_match(cn_word_emb, cn_word, cn_edge_start_time, cn_edge_duration, lat_words, lat_edge_start_times, lat_edge_durations, lat_posteriors, max_time_diff):
    """ Iterate each arc in the lattice and find the arc which corresponds to the confusion network arc.
        Returning an arc index of -1 indicates that no matches were found.
    """
    candidate_list = []
    min_cost = math.inf
    for lat_arc_idx, lat_word in enumerate(lat_words):
        # Check that the words match
        if (cn_word_emb == lat_word).all():
            cost = cost_fn(cn_edge_start_time, cn_edge_duration, lat_edge_start_times[lat_arc_idx], lat_edge_durations[lat_arc_idx], max_time_diff)
            if cost >= 0 and cost <= min_cost:
                if cost == min_cost:
                    candidate_list.append((lat_arc_idx, cost, lat_posteriors[lat_arc_idx]))
                else:
                    # cost < min_cost
                    candidate_list = [(lat_arc_idx, cost, lat_posteriors[lat_arc_idx])]
                    min_cost = cost
    if not candidate_list:
        print('No matches found for CN arc {} with start time: {} and duration: {}'.format(cn_word, cn_edge_start_time, cn_edge_duration))
        LOGGER.info('No matches found for CN arc {} with start time: {} and duration: {}'.format(cn_word, cn_edge_start_time, cn_edge_duration))
        return -1
    if len(candidate_list) > 1 and isinstance(candidate_list, list):
        # Need to sort by posterior and choose the one with the highest posterior
        match = sorted(candidate_list, key=lambda x: x[2], reverse=True)[0][0]
    elif len(candidate_list) == 1 and isinstance(candidate_list, list):
        match = candidate_list[0][0]
    else:
        # Must be just a tuple - only one candidate
        match = candidate_list[0]
    # Return the index of the matching arc in the lattice
    return match

def check_match_quality(lat_edge, lat_start_time, cn_edge, cn_start_time, cn_file_path):
    lat_end_time = lat_start_time + lat_edge[DURATION_IDX]
    cn_end_time = cn_start_time + cn_edge[DURATION_IDX]

    if abs(lat_start_time - cn_start_time) > FRAME_PERIOD * 5 or abs(lat_end_time - cn_end_time) > FRAME_PERIOD * 5:
        LOGGER.info('{}\n\t\t\t\t\t\t Lattice start time: {} Lattice end time: {}\n\t\t\t\t\t\t Confnet start time: {} Confnet end time: {}'.format(cn_file_path, lat_start_time, lat_end_time, cn_start_time, cn_end_time))

def enrich_cn(file_name, cn_path, lat_path, output_dir, include_lm, include_am, grapheme, reverse_wordvec, time_threshold):
    print('Enriching: {}'.format(file_name))
    success = True
    # For each edge in the confusion network, the following information is contained:
    # edge_data = [EMBEDDING_LENGTH(50), duration(1), arc_posterior(1)]
    cn = np.load(cn_path)
    cn_edge_data = cn['edge_data']
    cn_start_times = cn['start_times']

    # For each edge in the lattice, the following information is contained:
    # edge_data = [EMBEDDING_LENGTH(50), duration(1), AM(1), LM(1), arc_posterior(1)]
    # start_times = [start_time(1)]
    # grapheme_data = [max_num_graphemes(5 x 5)]
    lat = np.load(lat_path)
    lat_edge = lat['edge_data']
    lat_edge_start_times = lat['start_times']
    lat_grapheme_data = lat['grapheme_data']

    # Only need to iteratively update new_cn_edge_data if we are adding features
    added_feature_count = num_added_features(include_am, include_lm)
    if added_feature_count > 0:
        new_cn_edge_data = np.empty((cn_edge_data.shape[0], cn_edge_data.shape[1] + added_feature_count))
        update_cn_edge_data = True
    else:
        new_cn_edge_data = cn_edge_data
        update_cn_edge_data = False

    if grapheme:
        if 'grapheme_data' in cn.keys():
            if not (cn['grapheme_data'].any() == None):
                raise Exception('The source lattices already contain grapheme information')
        new_cn_grapheme_data = np.empty((cn_edge_data.shape[0], lat_grapheme_data.shape[1], lat_grapheme_data.shape[2]))
    elif 'grapheme_data' in cn.keys():
        new_cn_grapheme_data = cn['grapheme_data']
    else:
        new_cn_grapheme_data = None

    # For each arc in the confusion network
    for cn_edge_idx, cn_edge in enumerate(cn_edge_data):
        cn_word_embedding = cn_edge[:WORD_EMBEDDING_LENGTH]

        if (cn_word_embedding == np.zeros_like(cn_word_embedding)).all():
            # A zero array means that the CN edge is for the !NULL arc
            # This is ignored anyway so just fill with zeros as required
            lat_arc_idx = -2
        else:
            if str(cn_word_embedding) in reverse_wordvec:
                word = reverse_wordvec[str(cn_word_embedding)]
            else:
                print('No word-vec in the embedding for: {}'.format(cn_edge))
                LOGGER.info('No word-vec in the embedding for: {}'.format(cn_edge))

            lat_arc_idx = find_match(
                cn_word_emb=cn_word_embedding,
                cn_word=word,
                cn_edge_start_time=cn_start_times[cn_edge_idx],
                cn_edge_duration=cn_edge[DURATION_IDX],
                lat_words=lat_edge[:, :WORD_EMBEDDING_LENGTH],
                lat_edge_start_times=lat_edge_start_times,
                lat_edge_durations=lat_edge[:, DURATION_IDX],
                lat_posteriors=lat_edge[:, POST_IDX],
                max_time_diff = time_threshold
            )
        if lat_arc_idx >= 0:
            # For all arc indices which indicate a match in the lattice:

            # Log warning if they barely match:
            check_match_quality(
                lat_edge=lat_edge[lat_arc_idx],
                lat_start_time=lat_edge_start_times[lat_arc_idx],
                cn_edge=cn_edge,
                cn_start_time=cn_start_times[cn_edge_idx],
                cn_file_path=cn_path
            )

            # Adding grapheme level information if a matching arc was found
            if grapheme:
                new_cn_grapheme_data[cn_edge_idx] = lat_grapheme_data[lat_arc_idx]

            # Adding word level information
            if update_cn_edge_data:
                new_features = []
                if include_am:
                    new_features.append(lat_edge[lat_arc_idx, AM_INDEX])
                if include_lm:
                    new_features.append(lat_edge[lat_arc_idx, LM_INDEX])
                new_cn_edge_data[cn_edge_idx] = np.concatenate((cn_edge, np.array(new_features)), axis=0)
        elif lat_arc_idx == -2:
            if grapheme:
                new_cn_grapheme_data[cn_edge_idx] = np.zeros_like(lat_grapheme_data[0])
            if update_cn_edge_data:
                new_features = [0] * num_added_features(include_am, include_lm)
                new_cn_edge_data[cn_edge_idx] = np.concatenate((cn_edge, np.array(new_features)), axis=0)
        else:
            # Break and flag the lattice as not enriched
            success = False
            # break

    # save multiple variables into one .npz file
    full_file_path = os.path.join(output_dir, file_name)
    if success:
        if new_cn_grapheme_data is not None:
            np.savez(full_file_path, topo_order=cn['topo_order'], child_2_parent=cn['child_2_parent'],
                    parent_2_child=cn['parent_2_child'], edge_data=new_cn_edge_data,
                    ignore=cn['ignore'], grapheme_data=new_cn_grapheme_data, start_times=cn['start_times'])
        else:
            np.savez(full_file_path, topo_order=cn['topo_order'], child_2_parent=cn['child_2_parent'],
                    parent_2_child=cn['parent_2_child'], edge_data=new_cn_edge_data,
                    ignore=cn['ignore'], start_times=cn['start_times'])
    return success

def main(args):

    global LOGGER
    LOGGER = utils.get_logger(args.verbose, log_file_name=os.path.join(args.output_dir, 'enrich_cn'))
    LOGGER.info('================= Process Start =================')

    if not args.lm and not args.am and not args.grapheme:
        raise Exception('Nothing to enrich - ensure that you have set sensible CLI arguments') 

    cn_set = set_of_processed_file_names(directory_to_search=args.confusion_network_dir)
    lat_set = set_of_processed_file_names(directory_to_search=args.lattice_dir)

    reverse_wordvec = np.load('reverse_wordvec.npy').item()

    for file_name in cn_set:
        if file_name not in lat_set:
            raise Exception('No matching lattice for the confusion network file {}'.format(file_name))
        lat_path = os.path.join(args.lattice_dir, file_name)
        cn_path = os.path.join(args.confusion_network_dir, file_name)
        success = enrich_cn(file_name, cn_path, lat_path, args.output_dir, args.lm, args.am, args.grapheme, reverse_wordvec, args.time_threshold)
        if not success:
            LOGGER.info('CN {} was not enriched'.format(cn_path))
    LOGGER.info('================= Process complete =================')

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Match each arc in a confusion network with the corresponding " + \
                  "arc in the corresponding lattice to extract grapheme information"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        help='The directory to save the enriched confusion networks.'
    )
    parser.add_argument(
        '-c', '--confusion-network-dir', type=str, required=True,
        help='Directory to find the processed confusion network *.npz files'
    )
    parser.add_argument(
        '-l', '--lattice-dir', type=str, required=True,
        help='Directory to find the processed lattice *.npz files'
    )
    parser.add_argument(
        '--LM', dest='lm', action='store_true', default=False,
        help='Include the language model score on the confusion network arc'
    )
    parser.add_argument(
        '--AM', dest='am', action='store_true', default=False,
        help='Include the acoustic model score on the confusion network arc'
    )
    parser.add_argument(
        '--grapheme', dest='grapheme', action='store_true', default=False,
        help='Include the grapheme information on the confusion network arc'
    )
    parser.add_argument(
        '-t', '--time-threshold', type=float, default=1,
        help='The maximum acceptable error to match arcs'
    )
    parser.add_argument(
        '--debug', dest='debug_mode', action='store_true', default=False,
        help='Run in debug mode - print the word on each arc'
    )
    parser.add_argument(
        '-r', '--reverse-wordvec', type=str, required=True,
        help='Path to reverse wordvec dictionary'
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Set logging level: ERROR, WARNING (-v), INFO (-vv) (default), DEBUG (-vvv)',
        action='count', default=2
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
