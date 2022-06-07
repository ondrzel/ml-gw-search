#! /usr/bin/env python

### Import modules
from argparse import ArgumentParser
import logging
import h5py
import numpy as np
import os.path
from tqdm import tqdm

def copy_attrs(in_obj, out_obj):
    for key, attr in in_obj.attrs.items():
        out_obj.attrs[key] = attr
    return

def main():
    parser = ArgumentParser(description="Strain whitening script to be used for development of submission to the MLGWSC-1 mock data challenge. Written by Ondrej Zelenka.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    parser.add_argument('inputfile', type=str, help="The path to the input data file. It should be real_noise_file.hdf pulled by the generate_data.py script.")
    parser.add_argument('outputfiles', type=str, nargs=2, help="The paths where to store both output files. The files must not exist.")
    parser.add_argument('-d', '--duration', type=float, default=1.e6, help="Minimal duration in seconds of the first output file. Default: 1.e6.")
    parser.add_argument('--compress', action='store_true', help="Compress the output file.")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Check existence of output file
    for outfilepath in args.outputfiles:
        if os.path.isfile(outfilepath) and not args.force:
            raise RuntimeError("Output file %s exists." % outfilepath)
        else:
            pass

    if args.compress:
        ds_write_kwargs = {'compression': 'gzip', 'compression_opts': 9, 'shuffle': True}
    else:
        ds_write_kwargs = {}

    with h5py.File(args.inputfile, 'r') as infile, h5py.File(args.outputfiles[0], 'w') as outfile1, h5py.File(args.outputfiles[1], 'w') as outfile2:
        total = sum([len(grp) for grp in infile.values()])
        in_detector_group = next(iter(infile.values()))
        segment_keys = list(in_detector_group.keys())
        durations = [in_detector_group[key].attrs['delta_t']*len(in_detector_group[key]) for key in segment_keys]

        perm = np.random.default_rng().permutation(len(segment_keys))
        keys1 = []
        duration1 = 0.
        for index in perm:
            keys1.append(segment_keys[index])
            duration1 += durations[index]
            if duration1 > args.duration:
                break

        copy_attrs(infile, outfile1)
        copy_attrs(infile, outfile2)
        with tqdm(desc='Processing individual datasets', disable=not args.verbose, ascii=True, total=total) as pbar:
            for detector_group_name, in_detector_group in infile.items():
                out_detector_group1 = outfile1.create_group(detector_group_name)
                out_detector_group2 = outfile2.create_group(detector_group_name)
                copy_attrs(in_detector_group, out_detector_group1)
                copy_attrs(in_detector_group, out_detector_group2)
                for segment_name, in_segment in in_detector_group.items():
                    segment_data = in_segment[()]
                    if segment_name in keys1:
	                    out_segment = out_detector_group1.create_dataset(segment_name, data=segment_data, **ds_write_kwargs)
                    else:
	                    out_segment = out_detector_group2.create_dataset(segment_name, data=segment_data, **ds_write_kwargs)
                    copy_attrs(in_segment, out_segment)
                    pbar.update(1)

if __name__=='__main__':
    main()
