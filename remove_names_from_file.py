import argparse
import os
import sys


def get_names_to_remove(path):
    with open(path, 'r') as remove_file:
        return remove_file.read().splitlines()

def remove_and_save(names_to_remove, subset_list_file):
    with open(subset_list_file, 'r') as subset_file:
        subset_list = subset_file.readlines()

    original_subset_size = len(subset_list)
    for path in list(subset_list):
        for name in names_to_remove:
            if name in path:
                subset_list.remove(path)
                names_to_remove.remove(name)

    assert len(subset_list) != original_subset_size, \
        'No paths removed!'

    reduced_dataset_path = os.path.join(str(subset_list_file) + '.reduced')
    with open(reduced_dataset_path, 'w') as reduced_subset_file:
        reduced_subset_file.write(''.join(subset_list))

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.

        Arguments:
            args_to_parse: CLI arguments to parse
    """
    description = "Quick and dirty script to remove a list of names from the subset file with paths to lattices / confnets."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-i', '--remove-list-path', type=str,
        help="Path to the file containing the list of names to remove."
    )
    parser.add_argument(
        '-f', '--file-list-path', type=str,
        help="Path to the file containing the list of paths to files in the subset."
    )
    args = parser.parse_args(args_to_parse)
    return args


def main(args):
    """ Primary entry point for the script. """
    name_list = get_names_to_remove(path=args.remove_list_path)
    remove_and_save(
        names_to_remove=name_list,
        subset_list_file=args.file_list_path
    )


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
