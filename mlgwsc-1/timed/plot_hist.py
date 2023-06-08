#! /usr/bin/env python

# Copyright 2023 Ondřej Zelenka

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
import numpy as np
import matplotlib.pyplot as plt
import os.path

def main():
    parser = ArgumentParser(description="Script to plot timing histograms of different parts of the network. Written by Ondřej Zelenka.")

    parser.add_argument('inputfile', type=str, help="The path to the input times file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the output histogram. The file must not exist.")
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help="Set the limits on the x-axis containing the times as `--xlim lower upper`. If set to `none` the default limits are used.")
    parser.add_argument('--x-bin', type=float, nargs=2, default=None, help="Set the limits on the histogram bins as `--x-bin lower upper`. If set to `none` the default limits are used.")
    parser.add_argument('--bins', type=int, default=plt.rcParams['hist.bins'], help="Number of equal-log-width bins to be used for the histogram. If set to `none` the default count is used.")

    parser.add_argument('--figsize', type=float, nargs=2, default=[10.24, 7.68], help="The size of the final plot in inches. Set as `--figsize width height`.")
    parser.add_argument('--dpi', type=int, default=100, help="The DPI at which to generate the figure.")
    parser.add_argument('--fontsize', type=int, default=16, help="The font size to use for labels and legends.")
    parser.add_argument('--no-tex', action='store_true', help="Do not use LaTeX to render text.")
    parser.add_argument('--rasterize', action='store_true', help="Rasterize the output.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    logging.info('Starting')

    ### Check existence of output file
    if os.path.isfile(args.outputfile) and not args.force:
        raise RuntimeError("Output file exists.")

    plt.rcParams.update({'text.usetex': not args.no_tex,
                         'figure.dpi': args.dpi,
                         'figure.figsize': args.figsize,
                         'font.size': args.fontsize})

    logging.info('Setting up figure')
    fig, ax = plt.subplots(rasterized=args.rasterize)
    ax.set_xscale('log')
    if not args.xlim is None:
        ax.set_xlim(args.xlim)
    ax.set_xlabel('$t\\, \\left[\\mathrm{s}\\right]$')

    labels = ['flattening', 'classifier', 'convolutional part']
    colors = ['C1', 'C2', 'C0']

    logging.info('Loading data')
    raw_data = np.loadtxt(args.inputfile)
    weights = raw_data[:, -1]
    norm_data = 1e-9*raw_data.T[np.array((1, 2, 0))]/np.expand_dims(weights, axis=0)
    logging.debug('Loaded %i times in each class, reweighted to %i' % (raw_data.shape[1], sum(weights)))

    logging.info('Binning the data')
    x_bin = (np.amin(norm_data), np.amax(norm_data)) if args.x_bin is None else args.x_bin
    bins = np.exp(np.linspace(*np.log(x_bin), args.bins+1))
    hists = [np.histogram(col, bins=bins, weights=weights) for col in norm_data]
    max_y = max([max(counts) for counts, bins in hists])

    power10_y = int(np.floor(np.log10(max_y)))
    base10_y = 10**power10_y
    ylabel = 'counts $\\left[10^{%i}\\right]$' % power10_y
    ax.set_ylabel(ylabel)

    logging.info('Plotting histograms')
    for (counts, bins), lab, color in zip(hists, labels, colors):
        ax.stairs(counts/base10_y, bins, color=color, label=lab)

    ax.grid()
    ax.legend(loc='upper left', ncol=3)

    logging.info('Saving to %s' % args.outputfile)
    fig.savefig(args.outputfile)

if __name__=='__main__':
    main()

