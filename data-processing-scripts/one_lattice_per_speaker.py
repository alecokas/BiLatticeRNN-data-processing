"""
    This script is to be used to generate a subtrain set from a train set. Generally the train set file with be named 'train.txt'.
    The expected format of the train set file is that it contains a list of paths.
    For example:
    /home/babel/BABEL_OP3_404/releaseB/exp-graphemic-pmn26/latrnn/data/lattice_mapped_0.1_prec/lattices/BPL404-92740-20141126-025242-ou_COXXXXX_0037124_0037453.npz
"""

import argparse


def get_list(path):
    with open(path, 'r') as input_file:
        return input_file.readlines()

def save_list_to_file(list_to_save, dst_name):
    with open(dst_name, 'w') as target_file:
        for item in list_to_save:
            target_file.write('{}'.format(item))

def main():
    parser = argparse.ArgumentParser(description='Read in a txt file containing paths to lattices (*.npz) and extract N paths for each speaker')
    parser.add_argument(
        '-f', '--file-to-process', required=True, type=str,
        help='The path to the txt file containing the dataset from which to extract file paths'
    )
    parser.add_argument(
        '-d', '--dst-name', required=True, type=str,
        help='The output file for the subset list'
    )
    parser.add_argument(
        '-n', '--num-lats-per-speaker', default=10, type=int,
        help='The number of paths to extract from each speaker'
    )
    args = parser.parse_args()

    input_file = args.file_to_process
    lattice_list = get_list(input_file)

    speaker_dict = {}
    subset_list = []
    for lat_path in lattice_list:
        speaker, call_id = lat_path.split('XXXXX')
        if not speaker in speaker_dict.keys():
            speaker_dict[speaker] = 1
        else:
            speaker_dict[speaker] += 1
        if speaker_dict[speaker] <= args.num_lats_per_speaker:
            subset_list.append('{}XXXXX{}'.format(speaker, call_id))

    save_list_to_file(subset_list, args.dst_name)

    print('Total number of recordings in training set: {}'.format(len(lattice_list)))
    print('Total number of recordings in subtrain set: {}'.format(len(subset_list)))

if __name__=='__main__':
    main()
