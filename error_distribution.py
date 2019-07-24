import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
from scipy import stats
import sys


def read_pickle(file_name):
    """ Load the pickle file
    """
    with (open(file_name, "rb")) as openfile:
        return pickle.load(openfile)

def difference_error_fn(x1, x2):
    return float(x1) - float(x2)

def curate_error_data(lattice_start_time, lattice_stop_time, cn_start_time, cn_stop_time):
    """ Return the error data with the start time, stop time, and duration errors making up the columns of the array. """
    start_time_error = difference_error_fn(cn_start_time, lattice_start_time)
    stop_time_error = difference_error_fn(cn_stop_time, lattice_stop_time)
    duration_error = difference_error_fn(float(cn_stop_time) - float(cn_start_time), float(lattice_stop_time) - float(lattice_start_time))
    return [start_time_error, stop_time_error, duration_error]

def get_errors(input_file):
    errors = []
    with open(input_file, 'r') as log_file:
    
        lat_marker = 'Lattice start time'
        cn_marker = 'Confnet start time:'
        for line in log_file:
            if lat_marker in line:
                lattice_regex_results = re.findall(r'[0-9]+.[0-9]+', line)
                lattice_start_time, lattice_stop_time = lattice_regex_results

                cn_line = next(log_file)
                if not cn_marker in cn_line:
                    raise Exception('Unexpected format. A lattice line should always be followed by a confnet line')
                cn_regex_results = re.findall(r'[0-9]+.[0-9]+', cn_line)
                cn_start_time, cn_stop_time = cn_regex_results
                # print(lattice_start_time, lattice_stop_time)
                # print(cn_start_time, cn_stop_time)

                error_list = curate_error_data(lattice_start_time, lattice_stop_time, cn_start_time, cn_stop_time)
                errors.append(error_list)
    
    return np.array(errors)

def save_statistics(error_array, target_file_name):
    # Remove file if it exists
    try:
        os.remove(target_file_name)
    except OSError:
        pass

    stats_dict = {}
    error_type = ['Start Time', 'End Time', 'Duration']

    for errors, error_type in zip(error_array, error_type):
        stats_dict[error_type] = stats.describe(errors)

    # Write 
    print(error_type)
    with open(target_file_name + '.pickle', 'wb') as tgt_file:
            pickle.dump(stats_dict, tgt_file, protocol=pickle.HIGHEST_PROTOCOL)

def plot_distributions(error_array, directory):
    error_type = ['Start Time', 'End Time', 'Duration']
    for i, (errors, error_type) in enumerate(zip(error_array, error_type)):
        print(errors)
        fig = plt.figure()
        n, bins, patches = plt.hist(x=errors, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85, density=True)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Error')
        plt.ylabel('Normalised Probability Mass Error')
        plt.title('Empirical Probability Mass Distribution of the {} errors'.format(error_type))
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.max(n))
        file_name = os.path.join(directory, 'distribution-{}'.format(i))
        plt.savefig(file_name, dpi=fig.dpi)

def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    error_array = get_errors(args.input_file)
    plot_distributions(np.transpose(error_array), args.output_dir)
    save_statistics(np.transpose(error_array), target_file_name=os.path.join(args.output_dir,'error-stats'))


def parse_arguments(args_to_parse):
    """ Parse the command line arguments.
    """
    description = "Determine statistics on the arc matching errors"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        help='The directory to save the error distribution information.'
    )
    parser.add_argument(
        '-i', '--input-file', type=str, required=True,
        help='The path to the log file from which to extract the statistics'
    )

    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)