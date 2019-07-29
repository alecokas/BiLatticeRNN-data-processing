import argparse
import numpy as np
import os
import sys
import torch


def load_A_matrix(model_path, optim_path):
    model = torch.load(model_path)
    optim_state = torch.load(optim_path)
    state_dict = model.load_state_dict(optim_state['state_dict'])
    print(state_dict)

def load_embedding_txt(embedding_path):
    with open(embedding_path, 'r') as emb_file:
        embedding_list = emb_file.readlines()[1:]

    embedding_dict = {}
    for line in embedding_list:
        split_line = line.split()
        embedding_dict[split_line[0]] = np.array(split_line[1:], dtype=np.float)

def grapheme_counts():
    pass

def grapheme_durations():
    pass

def get_grapheme_info(processed_dir):
    for root, dirs, names in os.walk(processed_dir):
        for name in names:
            if name.endswith('.npz'):
                file_name = os.path.join(root, name)
                processed_file = np.load(file_name)
                grapheme_array = processed_file['grapheme_data']
                print(grapheme_array.shape)
                print(grapheme_array[0])
                break

def main(args):
    model_path = os.path.join(args.exp_model_dir, 'model_best.pth')
    optim_path = os.path.join(args.exp_model_dir, 'best.pth')
    # load_A_matrix(model_path, optim_path)

    get_grapheme_info(args.processed_dir)


def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Explore which graphemes and features are most prominently used by the attention model"
    parser = argparse.ArgumentParser(description=description)

    # parser.add_argument(
    #     '-e', '--embedding-info', type=str, required=True,
    #     help='A file with a list of names to check.'
    # )
    # parser.add_argument(
    #     '-m', '--exp-model-dir', type=str, required=True,
    #     help='Directory where the model is saved'
    # )
    parser.add_argument(
        '-p', '--processed-dir', type=str, required=True,
        help='Directory to search for processed lattice/confnet/onebest *.npz files'
    )
    # parser.add_argument(
    #     '-o', '--output-file', type=str, required=True,
    #     help='File to save the names of all missing files'
    # )
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)