import argparse
import os
from random import shuffle
import sys


# DIR = '/home/babel/BABEL_OP3_404/releaseB/exp-graphemic-pmn26/J2/decode-ibmseg-fcomb/test/dev'
# DIR = '/home/babel/BABEL_OP2_202/releaseB/exp-graphemic-ar527-v3/J1/decode-ibmseg-fcomb/test/dev/'
# TGT = 'swahili.dev.lst'
EXTENSION = '.scf.gz'


def prepare_file_names(names_list, validation_split, test_split):
    num_names = len(names_list)
    print('num_names: {}'.format(num_names))
    num_val_names = int(num_names * validation_split)
    num_test_names = int(num_names * test_split)
    num_train_names = num_names - num_val_names - num_test_names

    test = names_list[:num_test_names]
    val = names_list[num_test_names:num_test_names + num_val_names]
    train = names_list[num_test_names + num_val_names:]

    assert len(train) == num_train_names and len(val) == num_val_names and len(test) == num_test_names, \
        'Inconsistent lengths'

    assert num_names == num_train_names + num_val_names + num_test_names, \
        'Not all files included'

    subset_names_list = [train, val, test]
    lst_names = ['train.lst', 'cv.lst', 'test.lst']

    return subset_names_list, lst_names

def save(target, names_list):
    with open(target, 'w') as tgt_file:
        tgt_file.write('\n'.join(names_list))

def main(args):
    names_set = set()
    for root, dirs, names in os.walk(args.search_dir):
        for name in names:
            if name.endswith(EXTENSION):
                names_set.add(name[:-len(EXTENSION)])

    names_list = list(names_set)
    shuffle(names_list)
    subset_names_list, lst_names = prepare_file_names(names_list, args.validation, args.test)

    for subset_names, lst_name in zip(subset_names_list, lst_names):
        save(
            target=os.path.join(args.target_dest, lst_name),
            names_list=subset_names
        )

def parse_arguments(args_to_parse):
    """ Parse the command line arguments. """
    description = ""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-s', '--search-dir', type=str, required=True,
        help='The directory to search'
    )
    parser.add_argument(
        '-d', '--target-dest', type=str, required=True,
        help='The target directory to save the *.lst files'
    )
    parser.add_argument(
        '-v', '--validation', type=float, required=True,
        help='Ratio of the files to keep as the cross-validation set'
    )
    parser.add_argument(
        '-t', '--test', type=float, required=True,
        help='Ratio of the files to keep as the test set'
    )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
