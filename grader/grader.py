import click
from grader.histogram_helpers import *
from scipy.stats import wasserstein_distance
import csv


def score_chorale(chorale, dataset):
    """
    Arguments
        chorale: a music21 Stream object
        dataset: a ChoraleDataset object

    return score
    """
    assert dataset.histograms is not None
    note_score = get_note_score(chorale, dataset)
    rhythm_score = get_rhythm_score(chorale, dataset)

    return note_score, rhythm_score


def get_note_score(chorale, dataset):
    key = chorale.analyze('key')
    chorale_histogram = normalize_histogram(get_note_histogram(chorale, key))

    if key.mode == 'major':
        dataset_histogram = dataset.histograms['major_note_histogram']
    else:
        dataset_histogram = dataset.histograms['minor_note_histogram']

    chorale_list = [chorale_histogram[key] for key in dataset_histogram]
    dataset_list = [dataset_histogram[key] for key in dataset_histogram]

    return wasserstein_distance(chorale_list, dataset_list)


def get_rhythm_score(chorale, dataset):
    chorale_histogram = normalize_histogram(get_rhythm_histogram(chorale))
    dataset_histogram = dataset.histograms['rhythm_histogram']

    chorale_list = [chorale_histogram[key] for key in dataset_histogram]
    dataset_list = [dataset_histogram[key] for key in dataset_histogram]

    return wasserstein_distance(chorale_list, dataset_list)


def plot_distributions(chorale_file, generation_file):
    """
    Arguments
        dict: dictionary of score list

    plots many distributions on one graph, to visualize relationship between distributions
    """
    chorale_scores = []
    generation_scores = []
    with open(chorale_file, 'r') as chorale_file:
        reader = csv.reader(chorale_file)
        for row in reader:
            chorale_scores.append(row[-1])

    with open(generation_file, 'r') as generation_file:
        reader = csv.reader(generation_file)
        for row in reader:
            generation_scores.append(row[-1])

    plt.hist(chorale_scores, label='real chorales')
    plt.hist(generation_scores, label='generated chorales')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/score_distribution.png')
