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
import numpy as np
import h5py
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
import os, os.path
from tqdm import tqdm
import torch

from apply import get_coherent_network, get_coincident_network
from apply import dtype

### Basic dataset class for easy PyTorch loading
class Dataset(torch.utils.data.Dataset):
    def __init__(self, noises=None, waveforms=None,
                store_device='cpu', train_device='cpu',
                snr_range=(5., 15.)):
        torch.utils.data.Dataset.__init__(self)
        self.noises = noises
        self.waveforms = waveforms
        self.store_device = store_device
        self.train_device = train_device
        if not self.noises is None:
            self.convert()
        self.rng = np.random.default_rng()
        self.snr_range = snr_range
        self.wave_label = torch.tensor([1., 0.]).to(dtype=dtype, device=self.train_device)
        self.noise_label = torch.tensor([0., 1.]).to(dtype=dtype, device=self.train_device)
        return

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, i):
        if i<len(self.waveforms):
            snr = self.rng.uniform(*self.snr_range)
            return (self.noises[i]+snr*self.waveforms[i]).to(device=self.train_device), self.wave_label
        else:
            return self.noises[i].to(device=self.train_device), self.noise_label

    def convert(self):
        self.noises = torch.from_numpy(self.noises).to(dtype=dtype, device=self.store_device)
        self.waveforms = torch.from_numpy(self.waveforms).to(dtype=dtype, device=self.store_device)

    def save(self, h5py_file, group_name):
        if group_name in h5py_file.keys():
            raise IOError("Group '%s' in file already exists." % group_name)
        else:
            new_group = h5py_file.create_group(group_name)
            new_group.create_dataset('waveforms', data=self.waveforms.cpu().numpy())
            new_group.create_dataset('noises', data=self.noises.cpu().numpy())

    def load(self, h5py_file, group_name):
        if group_name in h5py_file.keys():
            group = h5py_file[group_name]
            self.noises = group['noises'][()]
            self.waveforms = group['waveforms'][()]
            self.convert()
        else:
            raise IOError("Group '%s' in file doesn't exist." % group_name)

### Basic dataset class for easy PyTorch loading
### Outdated class with fixed injections
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, samples=None, labels=None,
#                 store_device='cpu', train_device='cpu'):
#         torch.utils.data.Dataset.__init__(self)
#         self.samples = samples
#         self.labels = labels
#         self.store_device = store_device
#         self.train_device = train_device
#         if not self.samples is None:
#             self.convert()
#         return
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, i):
#         sample = self.samples[i].to(device=self.train_device)
#         label = self.labels[i].to(device=self.train_device)
#         return sample, label

#     def convert(self):
#         self.samples = torch.from_numpy(self.samples).to(dtype=dtype, device=self.store_device)
#         self.labels = torch.from_numpy(self.labels).to(dtype=dtype, device=self.store_device)
#         assert len(self.samples)==len(self.labels)

#     def save(self, h5py_file, group_name):
#         if group_name in h5py_file.keys():
#             raise IOError("Group '%s' in file already exists." % group_name)
#         else:
#             new_group = h5py_file.create_group(group_name)
#             new_group.create_dataset('samples', data=self.samples.cpu().numpy())
#             new_group.create_dataset('labels', data=self.labels.cpu().numpy())

#     def load(self, h5py_file, group_name):
#         if group_name in h5py_file.keys():
#             group = h5py_file[group_name]
#             self.samples = group['samples'][()]
#             self.labels = group['labels'][()]
#             self.convert()
#         else:
#             raise IOError("Group '%s' in file doesn't exist." % group_name)


class reg_BCELoss(torch.nn.BCELoss):
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
    def forward(self, inputs, target, *args, **kwargs):
        assert inputs.shape[-1]==self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*inputs
        return torch.nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)


def train(Network, training_dataset, validation_dataset, output_training,
          batch_size=32, learning_rate=5e-5, epochs=100,
          clip_norm=100, verbose=False, force=False):
    """Train a network on given data.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to train.
    training_dataset : (np.array, np.array)
        The data to use for training. The first entry has to contain the
        input data, whereas the second entry has to contain the target
        labels.
    validation_dataset : (np.array, np.array)
        The data to use for validation. The first entry has to contain
        the input data, whereas the second entry has to contain the
        target labels.
    output_training : str
        Path to a directory in which the loss history and the best
        network weights will be stored.
    weights_path: str
        Path where the trained network weights will be stored.
    batch_size : {int, 32}
        The mini-batch size used for training the network.
    learning_rate : {float, 5e-5}
        The learning rate to use with the optimizer.
    epochs : {int, 100}
        The number of full passes over the training data.
    clip_norm : {float, 100}
        The value at which to clip the gradient to prevent exploding
        gradients.
    verbose : {bool, False}
        Print update messages.
    force : {bool, False}
        Overwrite existing output files.
    
    Returns
    -------
    network
    """
    ### Set up data loaders as a PyTorch convenience
    logging.debug("Setting up datasets and data loaders.")
    TrainDL = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    ValidDL = torch.utils.data.DataLoader(validation_dataset, batch_size=500, shuffle=True)

    ### Initialize loss function, optimizer and output file
    logging.debug("Initializing loss function, optimizer and output file.")
    loss = reg_BCELoss(dim=2)
    opt = torch.optim.Adam(Network.parameters(), lr=learning_rate)

    losses_path = os.path.join(output_training, 'losses.txt')
    best_dict_path = os.path.join(output_training, 'best_state_dict.pt')
    if os.path.isfile(losses_path) and not force:
        raise RuntimeError("Output file %s exists." % losses_path)
    with open(losses_path, 'w', buffering=1) as outfile:
        ### Training loop
        best_loss = 1.e10 # impossibly bad value
        for epoch in tqdm(range(1, epochs+1), desc="Optimizing network", disable=not verbose, ascii=True):
            # Training epoch
            Network.train()
            training_running_loss = 0.
            training_batches = 0
            for training_samples, training_labels in tqdm(TrainDL, desc="Iterating over training dataset", leave=False, disable=not verbose, ascii=True):
                # Optimizer step on a single batch of training data
                opt.zero_grad()
                training_output = Network(training_samples)
                training_loss = loss(training_output, training_labels)
                training_loss.backward()
                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=clip_norm)
                # Make the actual optimizer step and save the batch loss
                opt.step()
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1
            # Evaluation on the validation dataset
            Network.eval()
            with torch.no_grad():
                validation_running_loss = 0.
                validation_batches = 0
                for validation_samples, validation_labels in tqdm(ValidDL, desc="Computing validation loss", leave=False, disable=not verbose, ascii=True):
                    # Evaluation of a single validation batch
                    validation_output = Network(validation_samples)
                    validation_loss = loss(validation_output, validation_labels)
                    validation_running_loss += validation_loss.clone().cpu().item()
                    validation_batches += 1
            # Print information on the training and validation loss in the current epoch and save current network state
            validation_loss = validation_running_loss/validation_batches
            output_string = '%04i    %f    %f' % (epoch, training_running_loss/training_batches, validation_loss)
            outfile.write(output_string + '\n')
            # Save 
            new_dict_path = os.path.join(output_training, 'state_dict_e_%04i.pt' % epoch)
            if os.path.isfile(new_dict_path) and not force:
                raise RuntimeError("Output file %s exists." % new_dict_path)
            torch.save(Network.state_dict(), new_dict_path)
            if validation_loss<best_loss:
                torch.save(Network.state_dict(), best_dict_path)
                best_loss = validation_loss

        logging.debug(("Training complete with best validation loss "
                        "%f, closing losses output file." % best_loss))
    return Network



def main():
    parser = ArgumentParser(description="CNN training script for submission to the MLGWSC-1 mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    parser.add_argument('-d', '--dataset-file', type=str, nargs='+', help="Path to the file where the datasets are stored.")
    parser.add_argument('-o', '--output-training', type=str, help="Path to the directory where the outputs will be stored. The directory must exist.")
    parser.add_argument('-s', '--snr', type=float, nargs=2, default=(5., 15.), help="Range from which the optimal SNRs will be drawn. Default: 5. 15.")
    parser.add_argument('-w', '--weights', help="The path to the file containing the initial network weights. If empty, the weights are initialized randomly.")
    parser.add_argument('--coincident', action='store_true', help="Train a set of networks for a coincident search. If omitted, a coherent search is trained.")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="Learning rate of the optimizer. Default: 0.00001")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size of the training algorithm. Default: 32")
    parser.add_argument('--clip-norm', type=float, default=100., help="Gradient clipping norm to stabilize the training. Default: 100.")
    parser.add_argument('--train-device', type=str, default='cpu', help="Device to train the network. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
    parser.add_argument('--store-device', type=str, default='cpu', help="Device to store the datasets. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Load datasets
    TrainDS_list = []
    ValidDS_list = []
    for infname in args.dataset_file:
        logging.debug("Loading datasets from %s." % infname)
        TrainDS = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        ValidDS = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        with h5py.File(infname, 'r') as dataset_file:
            TrainDS.load(dataset_file, 'training')
            ValidDS.load(dataset_file, 'validation')
        TrainDS_list.append(TrainDS)
        ValidDS_list.append(ValidDS)
    TrainDS = torch.utils.data.ConcatDataset(TrainDS_list)
    ValidDS = torch.utils.data.ConcatDataset(ValidDS_list)
    logging.debug("Datasets loaded.")

    ### Initialize network
    logging.debug("Initializing network.")
    if args.coincident:
        Network = get_coincident_network(path=args.weights, device=args.train_device, detectors=TrainDS[0][0].shape[0])
    else:
        Network = get_coherent_network(path=args.weights, device=args.train_device, detectors=TrainDS[0][0].shape[0])

    ### Train
    Network = train(Network, TrainDS, ValidDS, args.output_training,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    epochs=args.epochs, clip_norm=args.clip_norm,
                    verbose=args.verbose, force=args.force)

if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
