import argparse
import numpy as np
import sys


def parse_arguments(args_to_parse):
    description = "Process embeddings into a *.npy archive"
    parser = argparse.ArgumentParser(description=description)
    general = parser.add_argument_group('General options')
    general.add_argument(
        '-v', '--vector-file', type=str, required=True,
        help="The word embedding vector file (*.vec) as generated by fastText"
    )
    general.add_argument(
        '-s', '--save-path', type=str, required=True,
        help="The file path to save the word vector into *.npy format"
    )
    args = parser.parse_args(args_to_parse)
    return args

def read_vec(path):
    """ Read in the vector embedding generated by fastText and build a dictionary. """
    wordvec = {}
    with open(path, 'r') as vec_file:
        contents = vec_file.readlines()
    for line in contents[1:]:
        split_line = line.split()
        word = split_line[0]
        embedding = [float(elem) for elem in split_line[1:]]
        wordvec[word] = np.array(embedding)
    return wordvec

def main(args):
    wordvec = read_vec(path=args.vector_file)
    np.save(args.save_path, wordvec)

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)