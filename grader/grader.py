from grader.histogram_helpers import *
from scipy.stats import wasserstein_distance
import csv
import numpy as np
import matplotlib.pyplot as plt


def score_chorale(chorale, dataset, weights=None):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    return score
    """
    assert dataset.histograms is not None
    note_score = get_note_score(chorale, dataset)
    rhythm_score = get_rhythm_score(chorale, dataset)

    if weights is None:
        weights = [1]*(len(dataset.histograms)-1)
    scores = [note_score, rhythm_score]
    score = np.dot(weights, scores)

    return score, scores


def get_note_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
    """
    key = chorale.analyze('key')
    chorale_histogram = normalize_histogram(get_note_histogram(chorale, key))

    if key.mode == 'major':
        dataset_histogram = dataset.histograms['major_note_histogram']
    else:
        dataset_histogram = dataset.histograms['minor_note_histogram']

    chorale_list, dataset_list = get_ordered_histograms(chorale_histogram, dataset_histogram)

    return wasserstein_distance(chorale_list, dataset_list)


def get_rhythm_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
    """
    chorale_histogram = normalize_histogram(get_rhythm_histogram(chorale))
    dataset_histogram = dataset.histograms['rhythm_histogram']

    chorale_list, dataset_list = get_ordered_histograms(chorale_histogram, dataset_histogram)

    return wasserstein_distance(chorale_list, dataset_list)


def get_interval_score(chorale, dataset, directed=True):
    directed_ih, undirected_ih = normalize_histogram(get_interval_histogram(chorale))

    chorale_histogram = directed_ih if directed else undirected_ih
    key = 'directed_interval_histogram' if directed else 'undirected_interval_histogram'
    dataset_histogram = dataset.histograms[key]

    chorale_list, dataset_list = get_ordered_histograms(chorale_histogram, dataset_histogram)

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
            chorale_scores.append(float(row[1]))

    with open(generation_file, 'r') as generation_file:
        reader = csv.reader(generation_file)
        for row in reader:
            generation_scores.append(float(row[1]))

    plt.hist(chorale_scores, label='real chorales', alpha=0.5)
    plt.hist(generation_scores, label='generated chorales', alpha=0.5)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/score_distribution.png')
