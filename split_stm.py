#!/usr/bin/env python3
"""Script for spliting `.stm` files according to their names."""

import argparse
import numpy as np
import os
import utils

def get_segments(file_name):
    """Get segments names and number of lines for each segment."""
    count = 1
    total_num_lines = num_lines_in_file(file_name)
    with open(file_name, 'r') as file_in:
        pre_segment =  file_in.readline().split()[0]
        segments = [pre_segment]
        num_lines = []
        for line in file_in:
            line = line.split()
            if line[0].startswith(';;'):
                count += 1
            else:
                if len(line) > 5:
                    if line[0] == pre_segment:
                        count += 1
                    else:
                        segments.append(line[0])
                        pre_segment = line[0]
                        num_lines.append(count)
                        count = 1
                else:
                    # num_lines.append(count)
                    count += 1
        last_num_lines_entry = total_num_lines - sum(num_lines)
        num_lines.append(last_num_lines_entry)
    assert len(segments) == len(num_lines), "%i != %i" %(len(segments), len(num_lines))
    return segments, num_lines

def split_stm(file_name, segments, num_lines, dst_dir):
    """Split stm into multiple smaller ones by segments."""
    with open(file_name, 'r') as file_in:
        counter = 0
        for segment, num_line in zip(segments, num_lines):
            dst_file = os.path.join(dst_dir, segment + '.npz')
            time = []
            duration = []
            word = []
            for _ in range(num_line):
                line = next(file_in).split()
                counter += 1
                if line[0] != ';;': 
                    assert line[0] == segment, "Mismatch between {} and {}".format(line[0], segment)
                    time.append(float(line[3]))
                    duration.append(float(line[4]) - float(line[3]))
                    word.append(line[5])
            np.savez(dst_file, time=time, duration=duration, word=word)

def num_lines_in_file(file_name):
    """ Count the number of lines in a file, return num lines is zero if the file is empty """
    line_idx = -1
    with open(file_name) as file:
        for line_idx, _ in enumerate(file):
            pass
    return line_idx + 1

def main():
    """Main function for spliting `.stm` into `.npz` segnemts."""
    parser = argparse.ArgumentParser(
        description='Split the two *.stm alignment files into the respective lattice file directories'
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        help='Destination directory for the split output files'
    )
    parser.add_argument(
        '-dev', '--dev-stm', type=str, required=True,
        help='Path to the file containing the dev set train.stm alignment'
    )
    parser.add_argument(
        '-eval', '--eval-stm', type=str, required=True,
        help='Path to the file containing the eval set train.stm alignment'
    )
    args = parser.parse_args()

    if os.path.isfile(args.dev_stm) and os.path.isfile(args.eval_stm):
        stm_files = [args.dev_stm, args.eval_stm]
    else:
        raise FileNotFoundError('Please ensure that both {} and {} are valid files'.format(args.dev_stm, args.eval_stm))

    utils.mkdir(args.output_dir)

    for stm_file in stm_files:
        segments, num_lines = get_segments(stm_file)
        split_stm(stm_file, segments, num_lines, args.output_dir)

if __name__ == '__main__':
    main()
