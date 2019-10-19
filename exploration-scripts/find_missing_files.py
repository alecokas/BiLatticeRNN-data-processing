import argparse
import os
import sys

sys.path.append('../data-processing-scripts/')
from enrich_confusion_network import set_of_processed_file_names


def find_non_existent_files(path_list, remove_extension=True, extension='.npz'):
    """ Return a list of all file names which belong to files which do not exist """
    missing_files = []
    for path_name in path_list:
        if not os.path.isfile(path_name):
            if remove_extension:
                name = path_name.split('/')[-1]
                name = name[:-len(extension)]
                missing_files.append(name)
            else:
                missing_files.append(path_name)
    return missing_files

def files_to_search_for(file_name):
    with open(file_name, 'r') as list_file:
        content_list = list_file.readlines()
        return [elem.strip() for elem in content_list]

def main(args):
    """ Primary entry point for script which finds missing files """
    if args.full_paths:
        find_non_existent_files(args.list_file)
    else:
        files_in_dir = set_of_processed_file_names(
            directory_to_search=args.directory_to_search,
            remove_extension=True
        )
        query_files = files_to_search_for(args.list_file)

        missing_files = []
        for query_file in query_files:
            if not query_file in files_in_dir:
                missing_files.append(query_file)

    with open(args.output_file, 'w') as out_file:
        out_file.write('\n'.join(missing_files))

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Find missing files in a directory"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-l', '--list-file', type=str, required=True,
        help='A file with a list of names to check (*.lst) or (*.npz)'
    )
    parser.add_argument(
        '-d', '--directory-to-search', type=str, required=True,
        help='Directory to search for missing *.npz files'
    )
    parser.add_argument(
        '-o', '--output-file', type=str, required=True,
        help='File to save the names of all missing files'
    )
    parser.add_argument(
        '--full-paths', dest='full_paths', action='store_true', default=False,
        help='Flag to indicate that the list file contains the full paths to the search files'
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
