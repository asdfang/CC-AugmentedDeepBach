import sys
sys.path.insert(0, '../')

import csv, os
import matplotlib.pyplot as plt, numpy as np
from DeepBach.helpers import *
from music21 import converter
from grader.grader import score_chorale

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.helpers import *


weights = {'error': .5,
           'parallel_error': .15,
           'note': 5,
           'rhythm': 1,
           'directed_interval': 20}


def main():
    plot_score_per_iteration('../plots/')


def plot_distributions(chorale_file,
                       generation_file,
                       plot_dir,
                       col=1):
    """
    Arguments
        chorale_file: csv of real chorale scores
        generation_file: csv of generated scores
        plot_dir: what directory you want to name where your plots are going

    plots many distributions on one graph, to visualize relationship between distributions
    """
    chorale_scores = []
    generation_scores = []
    ensure_dir(plot_dir)
    with open(chorale_file, 'r') as chorale_file:
        reader = csv.reader(chorale_file)
        for i, row in enumerate(reader):
            if i == 0:
                title = row[col]
                continue
            chorale_scores.append(2-np.min([float(row[col]), 2]))

    with open(generation_file, 'r') as generation_file:
        reader = csv.reader(generation_file)
        for i, row in enumerate(reader):
            if i == 0:
                assert row[col] == title
                continue
            generation_scores.append(2 - np.min([float(row[col]), 2]))

    # plot distributions
    plt.figure()
    bins = np.histogram(np.hstack((chorale_scores, generation_scores)), bins=30)[1]
    plt.hist(chorale_scores, label='Real chorales', alpha=0.5, bins=bins)
    plt.hist(generation_scores, label='Generated chorales', alpha=0.5, bins=bins)
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.title('Score distribution for real and generated chorales (distributions)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'combined_dist_{title}.png'))

    # plot boxplots
    plt.figure()
    boxplot_data = [chorale_scores, generation_scores]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data)
    ax.set_xticklabels(['Real chorales', 'Generated chorales'])
    plt.ylabel(title)
    plt.title('Score distribution for real and generated chorales (boxplot)')
    plt.savefig(os.path.join(plot_dir, f'boxplots.png'))


def plot_score_per_iteration(dataset, plot_dir):
    """
    still in debugging mode
    """
    scores = []
    max_iteration = np.max([int(i) for i in os.listdir('generations/6/')])
    for iteration in range(max_iteration + 1):
        iter_scores = []
        for generation in os.listdir(f'generations/6/{iteration}/'):
            chorale = converter.parse(f'generations/6/{iteration}/{generation}')
            score = score_chorale(chorale, dataset, weights)[0]
            s = 2 - np.min([float(generation.split('_')[1][:-4]), 2])
            iter_scores.append(s)
        print(f'{iteration}: {len(iter_scores)}')
        assert len(iter_scores) == 50
        scores.append(iter_scores)

    plt.figure()
    boxplot_data = scores
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data)
    ax.set_xticklabels([str(i+1) for i in range(len(scores))])
    plt.ylabel('Score')
    plt.savefig(os.path.join(plot_dir, f'boxplots_per_iteration.png'))

