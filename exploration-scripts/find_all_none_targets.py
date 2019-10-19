import argparse
import numpy as np
import os
import sys


def find_anomalous_targets(directory_to_search):
    anomalous_target_list = []
    for root, _, names in os.walk(directory_to_search):
        for name in names:
            if name.endswith('.npz'):
                candidate_path = os.path.join(root, name)
                candidate_target = np.load(candidate_path).items()
                reference_key, reference = candidate_target[-1]
                if reference_key != 'ref':
                    print(candidate_target)
                    raise Exception('Expected to find the reference but found {}'.format(reference_key))
                if None in reference:
                    anomalous_target_list.append(name.split('.')[0])
    return anomalous_target_list

def main(args):
    """ Primary entry point for script which finds all targets with None in the reference one-best """

    anomalous_target_list = find_anomalous_targets(args.directory_to_search)
    with open(args.output_file, 'w') as out_file:
        out_file.write('\n'.join(anomalous_target_list))

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Find targets which contain a None reference"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-d', '--directory-to-search', type=str, required=True,
        help='Directory to search for anomylous target files'
    )
    parser.add_argument(
        '-o', '--output-file', type=str, required=True,
        help='File to save the names of all occurences'
    )

    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
