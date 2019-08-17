import argparse
import re
import sys


def check_for_arc(log_line):
    if 'No matches found for CN arc with' in log_line:
        return True
    else:
        return False

def get_arc(log_line):
    _, information = log_line.split('INFO')
    regex_results = re.findall(r'[0-9]+.[0-9]+', information)
    cn_start_time, cn_stop_time = regex_results
    return (cn_start_time, cn_stop_time)

def check_for_cn(log_line):
    if 'was not enriched' in log_line:
        return True
    else:
        return False

def get_cn(log_line):
    regex_results = re.findall(r'(!?.*\/)(.+)(.npz)', log_line)
    cn_name = regex_results[0][1]
    return cn_name

def main(args):
    missing_arc_list = []
    cn_arc_dict = {}
    with open(args.log_file, 'r') as log_file:
        for log_line in log_file:
            log_line = log_line.strip()
            if check_for_arc(log_line):
                missing_arc_list.append(get_arc(log_line))
            if check_for_cn(log_line):
                cn = get_cn(log_line)
                if not missing_arc_list:
                    raise Exception('Cannot have a CN fail without at least one arc fail: {}'.format(cn))
                cn_arc_dict[cn] = list(missing_arc_list)
                missing_arc_list = []

    print('Number of failed confusion networks: {}'.format(len(cn_arc_dict.keys())))
    total_num_arcs = 0
    for subset in ['train.txt', 'cv.txt', 'test.txt']:
        num_arcs = 0
        with open(subset, 'r') as subset_file:
            file_names = subset_file.readlines()
            for file_name in file_names:
                file_name = file_name.split('/')[-1].split('.')[0]
                if file_name in cn_arc_dict:
                    num_arcs += len(cn_arc_dict[file_name])
        subset_name = subset.split('.')[0]
        print('{} has {} arcs'.format(subset_name, num_arcs))
        total_num_arcs += num_arcs
    print('Total number of arcs: {}'.format(total_num_arcs))

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Generate a dictionary with each confusion network and its missing arcs"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-l', '--log-file', type=str, required=True,
        help='Log file with the CN arc enrichment errors'
    )
    parser.add_argument(
        '-s', '--subset-dir', type=str, required=True,
        help='Subset directory with the train, val, and test subset listings'
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
