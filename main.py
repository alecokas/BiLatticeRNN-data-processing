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

def compile_lattice_list(lst_destination,
                         base_directory='/home/dawna/babel/BABEL_OP3_404/releaseB/exp-graphemic-ar527-v3/J2/decode-ibmseg-fcomb/test/'):
    subsets = ['dev', 'eval']
    for subset in subsets:
        print(os.path.join(base_directory,subset))
        speaker_dirs = next(os.walk(os.path.join(base_directory,subset)))[1]
        for speaker_dir in speaker_dirs:
            raw_lattice_dir = os.path.join(speaker_dir, '/decode/rescore/tg_20.0_0.0/rescore/wlat_20.0_0.0/rescore/plat_20.0_0.0/lattices')
            raw_lattice_list = next(os.walk(raw_lattice_dir))[2]
            print(raw_lattice_list)
    

def main(args):
    """ Primary entry point for the script. """
    # unzip(
    #     input_dir=args.input_dir,
    #     target_directory=args.output_dir
    # )
    compile_lattice_list(None)



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
