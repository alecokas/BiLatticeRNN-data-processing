#!/usr/bin/env python3
"""A range of utility functions."""

import logging
import numpy as np
import os
import random
import re
import string
import sys

POSN_INFO_LEN = 2
APOSTROPHE_TOKEN = 'A'

def mkdir(directory):
    """Create directory if not exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            print("Direcotory %s already exists." %directory)

def savecmd(directory, cmd):
    """Cache commands."""
    cmddir = os.path.join(directory, 'CMDs')
    mkdir(cmddir)
    filename = os.path.splitext(cmd[0].split('/')[-1])[0] + '.cmds'
    cmdsep = '------------------------------------\n'
    with open(os.path.join(cmddir, filename), 'a+') as file:
        file.write(cmdsep)
        file.write(' '.join(cmd) + '\n')
        file.write(cmdsep)

def get_logger(level, log_file_name=None):
    """ Set logger object for stdout logging.
        Input: verbosity level specified in argument
        Return: logger object
    """
    infoformat = '%(asctime)s %(levelname)-7s %(message)s'
    dateformat = '%Y-%m-%d %H:%M'
    if log_file_name is not None:
        logging.basicConfig(level=logging.ERROR, format=infoformat, datefmt=dateformat,
                            filename='{}.log'.format(log_file_name), filemode='a+')
    else:
        logging.basicConfig(level=logging.ERROR, format=infoformat,
                            datefmt=dateformat)

    if level not in {0, 1, 2, 3}:
        raise ValueError('Unknown verbose level %d' % level)
    logger = logging.getLogger()
    logger.setLevel([logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG][level])
    return logger

def check_file(*args):
    """Check if a file exists."""
    for path in args:
        if not os.path.isfile(path):
            print("ERROR: File does not exist: %s" %path)
            sys.exit(1)

def check_dir(*args):
    """Check if a directory exists."""
    for path in args:
        if not os.path.isdir(path):
            print("ERROR: Directory does not exist: %s" %path)
            sys.exit(1)

def check_file_logging(logger, path):
    """Check if a file exists with logging."""
    if not os.path.isfile(path):
        logger.error("File does not exist: %s" %path)
        sys.exit(1)

def check_dir_logging(logger, path):
    """Check if a directory exists with logging."""
    if not os.path.isdir(path):
        logger.error("Directory does not exist: %s" %path)
        sys.exit(1)

def print_options(args):
    """Display all arguments, args is a object from argparse."""
    print_color_msg('==> All options are displayed below:')
    for arg in vars(args):
        print("".ljust(4) + "--{0:20}{1}".format(arg, getattr(args, arg)))

def randomword(length):
    """Generate random words."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def color_msg(msg):
    """Return colored message."""
    return "\033[38;5;108m%s\033[0m" %(msg)

def print_color_msg(msg):
    """Print colored message."""
    print("\033[38;5;108m%s\033[0m" %(msg))

def remove_file(file_path):
    """ Remove file if it exists """
    try:
        os.remove(file_path)
    except OSError:
        pass

def chunks(l, n):
    """ Yield successive n-sized chunks from list l. """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def append_to_file(item_to_append, target_file):
    with open(target_file, "a") as file:
        file.write(item_to_append + '\n')

def len_subword_features():
    """ TODO: There is probably a better way to centralize this """
    # Grapheme embedding (4), grapheme duration (1)
    LEN_GRAPHEME_FEATURES = 5
    return LEN_GRAPHEME_FEATURES

def get_grapheme_info(grapheme_info, subword_embedding_dict, apostrophe_embedding,
                      keep_pronunciation=True, uniform_durations=False):
    """ Extract grapheme information and store it in an array with the following form:
        ((emb-0-0, emb-0-1, emb-0-2, emb-0-3, dur-0)
            .       .         .        .       .
            .       .         .        .       .
            .       .         .        .       .
        (emb-J-0, emb-J-1, emb-J-2, emb-J-3, dur-J))
    """
    subword_list = grapheme_info.split(':')[1:-1]
    grapheme_feature_list = np.empty((len(subword_list), len_subword_features()))
    for i, subword_info in enumerate(subword_list):
        subword, subword_dur = subword_info.split(',')[:2]
        token = strip_subword(subword, 1, False, apostrophe_embedding, keep_pronunciation)
        if subword_embedding_dict is None:
            raise Exception('No subword embedding!')
        else:
            grapheme_feature_list[i, :] = np.append(subword_embedding_dict[token], subword_dur)
    if uniform_durations:
        # Sum and evenly distribute the duration
        word_dur = sum(grapheme_feature_list[:,-1])
        num_graphemes = grapheme_feature_list.shape[0]
        grapheme_feature_list[: -1] = word_dur / num_graphemes
    return grapheme_feature_list

def strip_subword(subword_info, subword_context_width, incl_posn_info, apostrophe_embedding, keep_pronunciation):
    """ Strip subwords of context and optionally the location indicator

        Arguments:
            subword_info: String with the full subword context information and location indicators.
            subword_context_width: The subword context width as an integer (the number of grams to consider)
            incl_posn_info: A boolean indicator for whether or not to include the subword position information (^I, ^M, ^F)
    """
    if subword_context_width > 3:
        raise Exception('The subword context width cannot be greater than 3.')

    itemised_subword_info = re.split(r'\+|\-', subword_info)
    if len(itemised_subword_info) == 1:
        subword = itemised_subword_info[0] if incl_posn_info else remove_location_indicator(itemised_subword_info[0], apostrophe_embedding)
    elif len(itemised_subword_info) == 3:
        if subword_context_width > 1:
            # Assume that if the context is 2 (bigram), we want the include the preceding subword unit
            stop = subword_context_width
            subword = ''.join(itemised_subword_info[:stop]) if incl_posn_info else remove_location_indicator(itemised_subword_info[:stop], apostrophe_embedding)
        else:
            subword = itemised_subword_info[1] if incl_posn_info else remove_location_indicator(itemised_subword_info[1], apostrophe_embedding)
    else:
        raise Exception('The subword unit length should be 1 or 3, but found {}'.format(len(itemised_subword_info)))
    return subword if keep_pronunciation else remove_pronunciation(subword)

def remove_location_indicator(subword_with_location, apostrophe_embedding):
    """ Strip location indicators from a string or strings within a list and return the result as a string

        Arguments:
            subword_with_location: Either a string or list containing the raw subword unit with location indicators.
    """
    if isinstance(subword_with_location, list):
        clean_subword_list = []
        for subword in subword_with_location:
            subword_split = subword.split('^')
            if len(subword_split) == 1:
                clean_subword_list.append(subword_split[0])
            else:
                clean_subword, apostrophe = clean_subword_split(subword_split)
                if apostrophe_embedding:
                    clean_subword_list.append(clean_subword)
                    if apostrophe:
                        clean_subword_list.append(apostrophe)
                else:
                    clean_subword_list.append(clean_subword + apostrophe)
        return ' '.join(clean_subword_list)
    else:
        subword_split = subword_with_location.split('^')
        if len(subword_split) == 1:
            return subword_split[0]
        else:
            clean_subword, apostrophe = clean_subword_split(subword_split)

            if apostrophe is not None:
                if apostrophe_embedding:
                    return ' '.join([clean_subword, apostrophe])
                else:
                    return ''.join([clean_subword, apostrophe])
            else:
                return clean_subword

def clean_subword_split(raw_subword_split):
    pronunciation = raw_subword_split[1][POSN_INFO_LEN - 1:]
    if pronunciation.endswith(APOSTROPHE_TOKEN):
        pronunciation = pronunciation.replace(APOSTROPHE_TOKEN, '')
        apostrophe = APOSTROPHE_TOKEN
    else:
        apostrophe = None

    raw_subword = raw_subword_split[0] + pronunciation
    return raw_subword, apostrophe

def remove_pronunciation(subword, delimiter=';'):
    """ Remove any pronunciation after the delimiter.
    """
    split_subword = subword.split(delimiter)
    if len(split_subword) == 1:
        return subword
    elif len(split_subword) > 2:
        raise Exception("Expected a maximum of one occurence of '{}'. Found {}".format(delimiter, len(split_subword)))
    return split_subword[0]

def load_wordvec(path):
    """Load pre-computed word vectors.

    Arguments:
        path {string} -- path to `.npy` file contains a dictionary of all words
            and their word vectors

    Returns:
        dictionary -- word vector
    """
    wordvec = np.load(path).item()
    return wordvec

def longest_grapheme_sequence(grapheme_list):
    """ Determine the length of the longest grapheme sequence in the provided list.
    
        Arguments:
            grapheme_list: Python list of the grapheme features
    """
    max_length_seq = -1
    for arc in grapheme_list:
        seq_length = arc.shape[0]
        if seq_length > max_length_seq:
            max_length_seq = seq_length
    if max_length_seq == -1:
        raise Exception('max_length never updated')
    return max_length_seq

def pad_subword_sequence(subword_seq, max_seq_length):
    """ The subword sequence (graphemic / phonetic) can be of variable length. In order to store
        this data in a numpy array, one pads and masks the subword dimension to the max sequence
        length.

        subword_seq: numpy array with dimensions (graphemes, features)
        max_seq_length: The length of the maximum subword sequence
    """
    pad_count = max_seq_length - subword_seq.shape[0]
    zero_pads = np.zeros((pad_count, len_subword_features()))
    padded_subword_seq = np.concatenate((subword_seq, zero_pads), axis=0)

    valid_array = np.ones_like(zero_pads, dtype=bool)
    invalid_array = np.zeros_like(subword_seq, dtype=bool)
    mask = np.concatenate((valid_array, invalid_array), axis=0)
    return padded_subword_seq, mask
