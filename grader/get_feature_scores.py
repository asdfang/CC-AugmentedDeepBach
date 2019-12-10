"""
score functions for a chorale with reference to a dataset
"""

from grader.histogram_helpers import *
from grader.compute_chorale_histograms import *
from scipy.stats import wasserstein_distance


def get_error_score(chorale, dataset):
    num_notes = len(chorale.flat.notes)
    chorale_histogram = get_error_histogram(chorale, dataset.voice_ranges)

    num_errors = sum(chorale_histogram.values())
    chorale_histogram = normalize_histogram(chorale_histogram)
    dataset_histogram = dataset.histograms['error_histogram']
    error_note_ratio = num_errors / num_notes

    return wasserstein_distance(*histogram_to_list(chorale_histogram, dataset_histogram)) * (error_note_ratio / dataset.error_note_ratio)

def get_parallel_error_score(chorale, dataset):
    num_notes = len(chorale.flat.notes)
    chorale_histogram = get_parallel_error_histogram(chorale)

    num_parallel_errors = sum(chorale_histogram.values())
    chorale_histogram = normalize_histogram(chorale_histogram)
    dataset_histogram = dataset.histograms['parallel_error_histogram']
    parallel_error_note_ratio = num_parallel_errors / num_notes

    return wasserstein_distance(*histogram_to_list(chorale_histogram, dataset_histogram)) * (parallel_error_note_ratio / dataset.parallel_error_note_ratio)


def get_note_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
    """
    key = chorale.analyze('key')
    chorale_histogram = normalize_histogram(get_note_histogram(chorale, key))
    dataset_histogram = dataset.histograms[f'{key.mode}_note_histogram']

    return wasserstein_distance(*histogram_to_list(chorale_histogram, dataset_histogram))


def get_rhythm_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
    """
    chorale_histogram = normalize_histogram(get_rhythm_histogram(chorale))
    dataset_histogram = dataset.histograms['rhythm_histogram']

    chorale_list, dataset_list = histogram_to_list(chorale_histogram, dataset_histogram)

    return wasserstein_distance(chorale_list, dataset_list)


def get_directed_interval_score(chorale, dataset):
    return _get_interval_score(chorale, dataset, directed=True)


def get_undirected_interval_score(chorale, dataset):
    return _get_interval_score(chorale, dataset, directed=False)


def _get_interval_score(chorale, dataset, directed=True):
    directed_ih, undirected_ih = get_interval_histogram(chorale)

    chorale_histogram = normalize_histogram(directed_ih) if directed else normalize_histogram(undirected_ih)
    key = 'directed_interval_histogram' if directed else 'undirected_interval_histogram'
    dataset_histogram = dataset.histograms[key]

    return wasserstein_distance(*histogram_to_list(chorale_histogram, dataset_histogram))


