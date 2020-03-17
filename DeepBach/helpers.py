"""
@author: Gaetan Hadjeres
"""

import torch
from torch.autograd import Variable
import re, os
import pickle
import numpy as np


def cuda_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def to_numpy(variable: Variable):
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def init_hidden(num_layers, batch_size, lstm_hidden_size, volatile=False):
    hidden = (
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size), volatile=volatile),
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size), volatile=volatile)
    )
    return hidden


def non_decreasing(list):
    return all(x <= y for x, y in zip(list, list[1:]))


def read_train_log(model_id):
    """
    used for debugging, read log file rather than re-training to record values
    """
    with open(f'logs/{model_id}_train_log.txt', 'r') as fin:
        rest = fin.read().replace('\n', '')
        for i in range(4):
            splits = rest.split(f'Training voice model {i+1}')
            if len(splits) == 2:
                curr_epoch, rest = splits
            else:
                curr_epoch = splits[0]
            train_loss_matches = re.findall("Training loss: (\d*\.\d*)", curr_epoch)
            train_loss = [float(l) for l in train_loss_matches]
            train_acc_matches = re.findall("Training accuracy: (\d*\.\d*)", curr_epoch)
            train_acc = [float(l) for l in train_acc_matches]
            val_loss_matches = re.findall("Validation loss: (\d*\.\d*)", curr_epoch)
            val_loss = [float(l) for l in val_loss_matches]
            val_acc_matches = re.findall("Validation accuracy: (\d*\.\d*)", curr_epoch)
            val_acc = [float(l) for l in val_acc_matches]

    loss_over_epochs = {'training': train_loss, 'validation': val_loss}
    acc_over_epochs = {'training': train_acc, 'validation': val_acc}
    return loss_over_epochs, acc_over_epochs


def ensure_dir(directory):
    """
    create directory if it does not already exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_or_pickle_distributions(dataset):
    distributions_file = 'grader/bach_distributions.txt'
    error_note_ratio_file = 'grader/error_note_ratio.txt'
    parallel_error_note_ratio_file = 'grader/parallel_error_note_ratio.txt'
    gaussian_file = 'grader/gaussian.txt'

    if os.path.exists(distributions_file) and os.path.exists(error_note_ratio_file) and os.path.exists(
            parallel_error_note_ratio_file) and os.path.exists(gaussian_file):
        print('Loading Bach chorale distributions')
        with open(distributions_file, 'rb') as fin:
            dataset.distributions = pickle.load(fin)
        with open(error_note_ratio_file, 'rb') as fin:
            dataset.error_note_ratio = pickle.load(fin)
        with open(parallel_error_note_ratio_file, 'rb') as fin:
            dataset.parallel_error_note_ratio = pickle.load(fin)
        with open(gaussian_file, 'rb') as fin:
            dataset.gaussian = pickle.load(fin)
    else:
        dataset.calculate_distributions()
        with open(distributions_file, 'wb') as fo:
            pickle.dump(dataset.distributions, fo)
        with open(error_note_ratio_file, 'wb') as fo:
            pickle.dump(dataset.error_note_ratio, fo)
        with open(parallel_error_note_ratio_file, 'wb') as fo:
            pickle.dump(dataset.parallel_error_note_ratio, fo)
        with open(gaussian_file, 'wb') as fo:
            pickle.dump(dataset.gaussian, fo)


def get_threshold(data_file=None, col=-1):
    thres = np.NINF         # minimum score seen so far
    
    with open(data_file, 'r') as fin:
        for row in fin:
            s = row.split(',')[col]
            if s > thres:
                thres = s
    return thres