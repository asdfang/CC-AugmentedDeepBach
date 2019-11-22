"""
@author: Gaetan Hadjeres
"""

import torch
from torch.autograd import Variable
import re


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


def read_train_log(file):
    """
    used for debugging, read log file rather than re-training to record values
    """
    with open(f'logs/{file}', 'r') as fin:
        log = fin.read()
        train_loss_matches = re.findall("Training loss: (\d*\.\d*)", log)
        train_loss = [float(l) for l in train_loss_matches]
        train_acc_matches = re.findall("Training accuracy: (\d*\.\d*)", log)
        train_acc = [float(l) for l in train_acc_matches]
        val_loss_matches = re.findall("Validation loss: (\d*\.\d*)", log)
        val_loss = [float(l) for l in val_loss_matches]
        val_acc_matches = re.findall("Validation accuracy: (\d*\.\d*)", log)
        val_acc = [float(l) for l in val_acc_matches]

    loss_over_epochs = {'training': train_loss, 'validation': val_loss}
    acc_over_epochs = {'training': train_acc, 'validation': val_acc}
    return loss_over_epochs, acc_over_epochs
