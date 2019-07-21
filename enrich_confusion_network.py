import argparse
import math
import numpy as np
import os
import sys
import utils


WORD_EMBEDDING_LENGTH = 50
DURATION_IDX = 50
AM_INDEX = 51
LM_INDEX = 52
POST_IDX = 53

def set_of_processed_file_names(directory_to_search):
    processed_names = set()
    for _, _, names in os.walk(directory_to_search):
        for name in names:
            if name.endswith('.npz'):
                processed_names.add(name)
    return processed_names

def num_added_features(include_am, include_lm):
    added_feature_count = 0
    if include_am:
        added_feature_count += 1
    if include_lm:
        added_feature_count += 1
    return added_feature_count

def cost_fn(x1_start_time, x1_dur, x2_start_time, x2_dur):
    x1_end_time = x1_start_time + x1_dur
    x2_end_time = x2_start_time + x2_dur
    return math.sqrt((x1_start_time - x2_start_time) ** 2 + (x1_end_time - x2_end_time) ** 2)

def find_match(cn_word, cn_edge_start_time, cn_edge_duration, lat_words, lat_edge_start_times, lat_edge_durations, lat_posteriors):
    # Iterate each arc in the lattice and find the arc which corresponds to the confusion network arc
    candidate_list = []
    min_cost = math.inf
    for lat_arc_idx, lat_word in enumerate(lat_words):
        # Check that the words match
        if (cn_word == lat_word).all():
            cost = cost_fn(cn_edge_start_time, cn_edge_duration, lat_edge_start_times[lat_arc_idx], lat_edge_durations[lat_arc_idx])
            if cost >= 0 and cost <= min_cost:
                if cost == min_cost:
                    candidate_list.append((lat_arc_idx, cost, lat_posteriors[lat_arc_idx]))
                else:
                    # cost < min_cost
                    candidate_list = (lat_arc_idx, cost, lat_posteriors[lat_arc_idx])
                    min_cost = cost
    if not candidate_list:
        print('No matches found')
        LOGGER.info('No matches found for CN arc with start time: {} and duration: {}'.format(cn_edge_start_time, cn_edge_duration))
    if len(candidate_list) > 1:
        # Need to sort by posterior and choose the one with the highest posterior
        match = sorted(candidate_list, key=lambda x: x[2], reverse=True)[0]
    else:
        match = candidate_list[0]
    # Return the index of the matching arc in the lattice
    return match[0]



def enrich_cn(file_name, cn_path, lat_path, output_dir, include_lm, include_am):
    print('Enriching: {}'.format(file_name))
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
    print('lat_edge.shape: {}'.format(lat_edge.shape))
    print('lat_edge_start_times.shape: {}'.format(lat_edge_start_times.shape))
    print('cn_edge_data.shape: {}'.format(cn_edge_data.shape))
    print('lat_grapheme_data.shape: {}'.format(lat_grapheme_data.shape))

    new_cn_edge_data = np.empty((cn_edge_data.shape[0], cn_edge_data.shape[1] + num_added_features(include_am, include_lm)))
    new_cn_grapheme_data = np.empty((cn_edge_data.shape[0], lat_grapheme_data.shape[1], lat_grapheme_data.shape[2]))

    # For each arc in the confusion network
    for cn_edge_idx, cn_edge in enumerate(cn_edge_data):
        lat_idx = find_match(
            cn_word=cn_edge[:WORD_EMBEDDING_LENGTH],
            cn_edge_start_time=cn_start_times,
            cn_edge_duration=cn_edge[DURATION_IDX],
            lat_words=lat_edge[:, :WORD_EMBEDDING_LENGTH],
            lat_edge_start_times=lat_edge_start_times,
            lat_edge_durations=lat_edge[:, DURATION_IDX],
            lat_posteriors=lat_edge[:, POST_IDX]
        )
        # Adding word level information
        if include_am or include_lm:
            new_features = []
            if include_am:
                new_features.append(lat_edge[lat_idx, AM_INDEX])
            if include_lm:
                new_features.append(lat_edge[lat_idx, LM_INDEX])
            new_cn_edge_data[cn_edge_idx] = np.concatenate((cn_edge, np.array(new_features)), axis=1)
        # Adding grapheme level information
        new_cn_grapheme_data[cn_edge_idx] = lat_grapheme_data[lat_idx]

    # save multiple variables into one .npz file
    full_file_path = os.path.join(output_dir, file_name)
    np.savez(full_file_path, topo_order=lat['topo_order'], child_2_parent=lat['child_2_parent'],
             parent_2_child=lat['parent_2_child'], edge_data=new_cn_edge_data,
             ignore=lat['ignore'], grapheme_data=new_cn_grapheme_data, start_times=lat['start_times'])


def main(args):

    global LOGGER
    LOGGER = utils.get_logger(args.verbose, log_file_name=os.path.join(args.output_dir, 'enrich_cn.log'))

    cn_set = set_of_processed_file_names(directory_to_search=args.confusion_network_dir)
    lat_set = set_of_processed_file_names(directory_to_search=args.lattice_dir)

    for file_name in cn_set:
        if file_name not in lat_set:
            raise Exception('No matching lattice for the confusion network file {}'.format(file_name))
        lat_path = os.path.join(args.lattice_dir, file_name)
        cn_path = os.path.join(args.confusion_network_dir, file_name)
        enrich_cn(file_name, cn_path, lat_path, args.output_dir, args.lm, args.am)



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
        '-v', '--verbose',
        help='Set logging level: ERROR, WARNING (-v), INFO (-vv) (default), DEBUG (-vvv)',
        action='count', default=2
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
