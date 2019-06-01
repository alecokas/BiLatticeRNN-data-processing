import argparse
import os
from subprocess import call
import sys


def parse_arguments(args_to_parse):
    return None
    """ Parse the command line arguments.

        Arguments:
            args_to_parse: CLI arguments to parse
    """
    description = "Extract information from phone marked lattices"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-i', '--input-dir', type=str,
        help="Path to the directory containing phone marked lattices."
    )
    parser.add_argument(
        '-o', '--output-dir', type=str,
        help="Output directory in which to save the uncompressed lattices and the extracted data."
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

def compile_lattice_list(destination='info/abs-dataset-paths',
                         base_directory='/home/dawna/babel/BABEL_OP3_404/releaseB/exp-graphemic-ar527-v3/J2/decode-ibmseg-fcomb/test/'):
    subsets = ['dev', 'eval']
    path_list = []
    for subset in subsets:
        subset_dir = os.path.join(base_directory, subset)
        speaker_dirs = next(os.walk(os.path.join(base_directory, subset)))[1]
        for speaker_dir in speaker_dirs:
            raw_lattice_dir = os.path.join(subset_dir, speaker_dir, 'decode/rescore/tg_20.0_0.0/rescore/wlat_20.0_0.0/rescore/plat_20.0_0.0/lattices')
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
    
def save_train_val_test_split(reference_dict, path_list):
    """
    """
    for dataset_name, lattice_set in reference_dict.items():
        dataset_list = []
        for idx, path in enumerate(list(path_list)):
            if path in lattice_set:
                dataset_list.append(path_list.pop(idx))
            print(len(path_list))
        

def save_txt_file(path_list, txt_file_name):
    # Remove file if it exists
    try:
        os.remove(txt_file_name)
    except OSError:
        pass
    # Write to new file
    with open(txt_file_name, 'a+') as txt_file:
        for path in path_list:
            txt_file.write(path + '\n')


def main(args):
    """ Primary entry point for the script. """
    # unzip(
    #     input_dir=args.input_dir,
    #     target_directory=args.output_dir
    # )
    reference_dict = read_reference_lst_files()
    path_list = compile_lattice_list()
    save_train_val_test_split(reference_dict, path_list)



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
