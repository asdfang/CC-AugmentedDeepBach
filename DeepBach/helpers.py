"""
@author: Gaetan Hadjeres
"""

import torch
from torch.autograd import Variable
import re, os


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


def init_hidden(num_layers, batch_size, lstm_hidden_size,
                volatile=False):
    hidden = (
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size),
            volatile=volatile),
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size),
            volatile=volatile)
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
