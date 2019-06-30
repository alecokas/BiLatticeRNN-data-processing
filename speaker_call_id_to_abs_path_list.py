""" Converts a *.lst file which contains the speaker call ID for each lattice to include in each of the
    subsets (training, cross-validation, testing) into a *.lat.txt file with the absolute paths to
    the lattices.
"""

import argparse
import os
import re
from subprocess import call
import sys


def parse_arguments(args_to_parse):
    """ Parse the command line arguments.

        Arguments:
            args_to_parse: CLI arguments to parse
    """
    description = "Extract information from phone marked lattices"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-b', '--base-lat-dir', type=str,
        help="Path to the base lattice directory."
    )
    parser.add_argument(
        '-e', '--extension-dir', type=str,
        help="Extension directory post-dataset directory"
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='info/abs-dataset-paths',
        help="Output directory for the processed absolute path files (.txt)"
    )  
    parser.add_argument(
        '-i', '--input-dir', type=str, default='info/reference-lists',
        help="The directory with the train, cv, and test files which indicate the dataset split (.lst)"
    )

    args = parser.parse_args(args_to_parse)
    return args

def unzip(input_dir, target_directory):
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.endswith('.gz'):
                input_path_name = os.path.join(root, name)
                try:
                    os.makedirs(target_directory)
                except FileExistsError:
                    pass
                call('zcat {} > {}'.format(input_path_name, os.path.join(target_directory, name[:-3])), shell=True)
            else:
                raise Exception('The target file must have the .gz extension.')

def compile_lattice_list(base_directory='/home/dawna/babel/BABEL_OP3_404/releaseB/exp-graphemic-ar527-v3/J2/decode-ibmseg-fcomb/test/',
                         ext_dir='decode/rescore/tg_20.0_0.0/rescore/wlat_20.0_0.0/rescore/plat_20.0_0.0/rescore/tg_lat_post_prec_20.0_0.0/lattices/'):
    subsets = ['dev', 'eval']
    path_list = []
    for subset in subsets:
        subset_dir = os.path.join(base_directory, subset)
        speaker_dirs = next(os.walk(os.path.join(base_directory, subset)))[1]
        for speaker_dir in speaker_dirs:
            raw_lattice_dir = os.path.join(subset_dir, speaker_dir, ext_dir)
            raw_lattice_list = next(os.walk(raw_lattice_dir))[2]
            for lattice_name in raw_lattice_list:
                abs_path = os.path.join(raw_lattice_dir, lattice_name)
                path_list.append(abs_path)
    return path_list


def read_reference_lst_files(dataset_split_dir='info/reference-lists'):
    """ Read in the files containing the reference for which lattices are in the
        train, cross-validation, and test sets. These are stored in 3 sets which
        can be indexed from a dictionary.
    """
    datasets = ['train.lst', 'cv.lst', 'test.lst']
    dataset_dict = {}
    for dataset in datasets:
        file_path = os.path.join(dataset_split_dir, dataset)
        file_set = set(line.strip() for line in open(file_path))
        dataset_name = dataset.split('.')[0]
        dataset_dict[dataset_name] = file_set
    return dataset_dict
    
def save_train_val_test_split(reference_dict, path_list, target_destination):
    """
    """
    lattice_name_list = []

    for abs_path in path_list:
        result = re.search(r'.*\/(.*)(\.lat\.gz)', abs_path)
        lattice_name_list.append(result.group(1))

    for dataset_name, lattice_set in reference_dict.items():
        dataset_list = [abs_path for name, abs_path in zip(lattice_name_list, path_list) if name in lattice_set]
        save_txt_file(dataset_list, os.path.join(target_destination, dataset_name))
        

def save_txt_file(path_list, txt_file_name):
    # Remove file if it exists
    try:
        os.remove(txt_file_name)
    except OSError:
        pass
    # Write to new file
    with open(txt_file_name + '.lat.txt', 'a+') as txt_file:
        for path in path_list:
            txt_file.write(path + '\n')


def main(args):
    """ Primary entry point for the script. """
    reference_dict = read_reference_lst_files(
        dataset_split_dir=args.input_dir
    )
    path_list = compile_lattice_list(
        base_directory=args.base_lat_dir,
        ext_dir=args.extension_dir
    )
    save_train_val_test_split(
        reference_dict=reference_dict,
        path_list=path_list,
        target_destination=args.output_dir,
    )


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
