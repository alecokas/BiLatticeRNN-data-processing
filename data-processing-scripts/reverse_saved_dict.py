""" Script for generating a reversed dictionary """

import argparse
import numpy as np
import sys


def parse_arguments(args_to_parse):
    description = "Load a *.npy archive of a dictionary and swap (reverse) the dictionary keys and values around"
    parser = argparse.ArgumentParser(description=description)
    general = parser.add_argument_group('General options')
    general.add_argument(
        '-i', '--input-file', type=str, required=True,
        help="The file path to the word vector dictionary into *.npy format"
    )
    general.add_argument(
        '-o', '--output-file', type=str, required=True,
        help="The target file to save the reversed dictionary"
    )
    args = parser.parse_args(args_to_parse)
    return args

def main(args):
    wordvec = np.load(args.input_file).items()
    reversed_wordvec = {v: k for k, v in wordvec.items()}
    np.save(args.output_file, reversed_wordvec)

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
