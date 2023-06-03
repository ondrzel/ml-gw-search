#! /usr/bin/env python

# Copyright 2022 Ondřej Zelenka

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

### Import modules
from argparse import ArgumentParser
import logging
import h5py
import numpy as np
import os.path
import pycbc.psd, pycbc.detector
from tqdm import tqdm

from apply import SegmentSlicer, regularize_psd

np_gen = np.random.default_rng()
delta_t = 1./2048.

class RealNoiseGetter:
    def __init__(self, fpath, detectors=None, step_size=0.5,
            slice_length=2048, low_frequency_cutoff=20.,
            segment_duration=0.5, max_filter_duration=0.25, pbar=None):
        self.detectors = [det.name for det in detectors]
        self.fpath = fpath
        self.step_size = step_size
        self.slice_length = slice_length
        self.low_frequency_cutoff = low_frequency_cutoff
        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        self.all_psds = []
        self.all_whitening_psds = []
        self.psd_index = -1
        self.psd_delta_f = 1./(delta_t*slice_length + max_filter_duration)
        self.pbar = pbar
        return
    def __iter__(self):
        self.seg_keys = None
        self.start_next_ds()
        return self
    def __next__(self):
        try:
            return self.get_next_slice()
        except StopIteration:
            if not self.pbar is None:
                self.pbar.update(1)
            self.start_next_ds()
            return self.get_next_slice()
    def start_next_ds(self):
        with h5py.File(self.fpath, 'r') as infile:
            if self.seg_keys is None:
                self.seg_keys = list(infile[self.detectors[0]].keys())
                logging.debug("Original seg_keys: %s" % self.seg_keys)
                np_gen.shuffle(self.seg_keys)
                logging.debug("Shuffled seg_keys: %s" % self.seg_keys)
                self.seg_keys_iter = iter(self.seg_keys)
            current_seg_key = next(self.seg_keys_iter)
            self.slicer = SegmentSlicer(infile, current_seg_key,
                    step_size=self.step_size, slice_length=self.slice_length,
                    detectors=self.detectors, white=False, save_psd=True,
                    low_frequency_cutoff=self.low_frequency_cutoff,
                    segment_duration=self.segment_duration, max_filter_duration=self.max_filter_duration)
            self.psds = [pycbc.psd.interpolate(psd, self.psd_delta_f).astype(np.float64) for psd in self.slicer.psds]
            self.whitening_psds = [regularize_psd(psd, self.low_frequency_cutoff) for psd in self.psds]
            self.all_psds.append(np.stack(self.psds, axis=0))
            self.all_whitening_psds.append(np.stack(self.whitening_psds, axis=0))
            self.slicer_iter = iter(self.slicer)
            self.psd_index += 1
    def get_next_slice(self):
        return next(self.slicer_iter)[0], self.psd_index
    def write_metadata(self, h5py_object, **kwargs):
        psds_dataset = h5py_object.create_dataset('psds', data=np.stack(self.all_psds, axis=0), **kwargs)
        psds_dataset.attrs['delta_f'] = self.psd_delta_f
        w_psds_dataset = h5py_object.create_dataset('whitening_psds', data=np.stack(self.all_whitening_psds, axis=0), **kwargs)
        w_psds_dataset.attrs['delta_f'] = self.psd_delta_f
        h5py_object.attrs['detectors'] = self.detectors

def write_data(filename_format, noises, psd_indices, chunk_index, noise_getter, force, **kwargs):
    out_fname = filename_format % chunk_index
    ### Check existence of output file
    if os.path.isfile(out_fname) and not force:
        raise RuntimeError("Output file exists.")
    else:
        pass
    filemode = 'w' if force else 'w-'
    
    with h5py.File(out_fname, filemode) as outf:
        outf.create_dataset('noises', data=np.stack(noises, axis=0), **kwargs)
        outf.create_dataset('psd_indices', data=np.array(psd_indices), **kwargs)
        noise_getter.write_metadata(outf, **kwargs)

def main():
    parser = ArgumentParser(description="Real noise slicing and whitening script for the MLGWSC-1 mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('-o', '--output-directory', type=str, help="Path to the directory where the datasets will be stored.")
    parser.add_argument('-d', '--detectors', type=int, default=1, help="Number of detectors.")
    parser.add_argument('-s', '--sample-length', type=int, default=2048, help="Number of values (length) of each sample. Sampling rate fixed at 2048 Hz. Default: 2048 (i.e. 1 s samples)")
    parser.add_argument('--chunk-size', type=int, default=20000, help="Size of each output file, for convenience. Default: 20 000")
    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")
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

    out_fname_format = os.path.join(args.output_directory, 'sliced_noise_%04i.hdf')

    ### Create the detectors
    detectors_abbr = ('H1', 'L1', 'V1', 'K1')[:args.detectors]
    detectors = []
    for det_abbr in detectors_abbr:
        detectors.append(pycbc.detector.Detector(det_abbr))

    sample_duration = args.sample_length*delta_t
    ### Whitening parameters
    segment_duration = 0.5
    max_filter_duration = 0.25

    with h5py.File(args.inputfile, 'r') as infile:
        num_keys = len(next(iter(infile.values())).keys())

    if args.compress:
        ds_write_kwargs = {'compression': 'gzip', 'compression_opts': 9, 'shuffle': True}
    else:
        ds_write_kwargs = {}

    noises = []
    psd_indices = []
    chunk_index = 0
    with tqdm(disable=not args.verbose, ascii=True, total=num_keys) as pbar:
        noise_getter = RealNoiseGetter(args.inputfile, detectors=detectors,
            slice_length=args.sample_length, low_frequency_cutoff=20.,
            segment_duration=segment_duration, max_filter_duration=max_filter_duration, pbar=pbar)
        for noise, psd_index in noise_getter:
            noises.append(noise)
            psd_indices.append(psd_index)
            if len(noises) >= args.chunk_size:
                write_data(out_fname_format, noises, psd_indices, chunk_index, noise_getter, args.force, **ds_write_kwargs)
                noises = []
                psd_indices = []
                chunk_index += 1
        if len(noises) > 0:
            write_data(out_fname_format, noises, psd_indices, chunk_index, noise_getter, args.force, **ds_write_kwargs)




if __name__=='__main__':
	main()
