import sys
import csv, os
from tqdm import tqdm
import matplotlib.pyplot as plt, numpy as np
from music21 import converter

sys.path[0] += '/../'
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
    for col in range(2, 9):
        plot_boxplot_per_iteration(data_file='data/22_update_scores_over_bach_chorales.csv', 
                                    plot_dir="plots/model_22", col=col)
    # plot_distributions(chorale_file='data/chorale_scores.csv',
    #                     generation_file='data/generation_scores.csv',
    #                     plot_dir='plots/',
    #                     filename='model_10_11.png',
    #                     plt_title='Score distribution of real and generated chorales',
    #                     col=1)


def plot_distributions(filename,
                       plt_title,
                       chorale_file='data/chorale_scores.csv',
                       generation_file='data/generation_scores.csv',
                       plot_dir='plots/',
                       col=1):
    """
    Arguments
        chorale_file: csv of real chorale scores
        generation_file: csv of generated scores
        plot_dir: what directory you want to name where your plots are going

    plots score distributions of real and generated chorales
    """
    chorale_scores = []
    generation_scores = []
    ensure_dir(plot_dir)
    with open(chorale_file, 'r') as chorale_file:
        reader = csv.reader(chorale_file)
        for i, row in enumerate(reader):
            if i == 0:
                label = row[col]
                continue
            chorale_scores.append(np.max([float(row[col]), -200]))

    with open(generation_file, 'r') as generation_file:
        reader = csv.reader(generation_file)
        for i, row in enumerate(reader):
            if i == 0:
                assert row[col] == label
                continue
            generation_scores.append(np.max([float(row[col]), -200]))

    # plot distributions
    plt.figure()
    bins = np.histogram(np.hstack((chorale_scores, generation_scores)), bins=100)[1]
    plt.hist(chorale_scores, label='Model 10 (base model, no transpositions)', alpha=0.5, bins=bins)
    plt.hist(generation_scores, label='Model 11 (DeepBach, includes transpositions)', alpha=0.5, bins=bins)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(os.path.split(plot_dir)[0], 'dist_' + os.path.split(plot_dir)[1]))

    # plot boxplots
    plt.figure()
    boxplot_data = [chorale_scores, generation_scores]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data)
    ax.set_xticklabels(['Real chorales', 'Generated chorales'])
    plt.ylabel(label)
    plt.title(title)
    plt.savefig(os.path.join(os.path.split(plot_dir)[0], 'boxplot_' + os.path.split(plot_dir)[1]))


def plot_boxplot_per_iteration(data_file, plot_dir='plots/', col=2):
    ensure_dir(plot_dir)
    label, scores = read_scores(data_file, col=col)
    plt.figure()
    # boxplot_data = []
    # for iter_scores in scores:
    #     iter_scores = [s for s in iter_scores if s >= -50]
    #     boxplot_data.append(iter_scores)
    boxplot_data = scores
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data)
    if col == 2:
        thres = get_threshold('data/chorale_scores.csv')
        plt.axhline(y=thres, color='steelblue', linestyle='-')
    ax.set_xticklabels([str(i) for i in range(len(scores))])
    plt.xlabel('Iteration')
    if col == 2:
        plt.ylabel('Grade')
    else:
        plt.ylabel('Distance')
    plt.title(f'{label} distribution of generations after each update iteration')
    plt.savefig(os.path.join(plot_dir, f'{label}_boxplots.png'))


def plot_selections_per_iteration(data_file, plot_dir='plots/'):
    """
    plot number of selections each iteration
    """
    thres = get_threshold('data/chorale_scores.csv')
    label, scores = read_scores(data_file, thres=thres)
    picked = [np.sum([1 for s in iter_scores if s > thres]) for iter_scores in scores]
    plt.figure()
    rects = plt.bar(range(1, len(picked)+1), picked)
    label_bars(rects)
    plt.xlabel('Iteration')
    plt.ylabel('Number of selected generations')
    plt.title('Number of selected generations in each update iteration')
    plt.savefig(os.path.join(plot_dir, '2_29_selections_per_iteration.png'))


def read_scores(data_file, col=2, thres=0):
    """
    read scores from CSV file
    """
    scores = []
    iter = 0
    with open(data_file, 'r') as fin:
        iter_scores = []
        for row_id, row in enumerate(fin):
            bits = row.split(',')
            if row_id == 0:
                label = bits[col].strip()
                continue
            i = int(bits[0])
            score = float(bits[col])
            if i != iter:
                scores.append(iter_scores)
                iter_scores = []
                iter += 1
            iter_scores.append(score)
        scores.append(iter_scores)
    return label, scores


def label_bars(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


if __name__ == '__main__':
    main()
