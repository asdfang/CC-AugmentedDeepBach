import click
from grader.histogram_helpers import *
from scipy.stats import wasserstein_distance


def score_chorale(chorale, dataset):
    """
    Arguments
        chorale: a music21 Stream object
        dataset: a ChoraleDataset object

    return score
    """
    assert dataset.histograms is not None

    key = chorale.analyze('key')
    chorale_histogram = normalize_histogram(get_note_histogram(chorale, key))

    if key.mode == 'major':
        dataset_histogram = dataset.histograms['major_note_histogram']
    else:
        dataset_histogram = dataset.histograms['minor_note_histogram']

    chorale_list = [chorale_histogram[key] for key in dataset_histogram]
    dataset_list = [dataset_histogram[key] for key in dataset_histogram]

    return wasserstein_distance(chorale_list, dataset_list)

