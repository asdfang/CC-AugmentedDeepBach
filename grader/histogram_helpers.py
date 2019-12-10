from collections import Counter
import csv, os
import matplotlib.pyplot as plt, numpy as np
from DeepBach.helpers import *


def normalize_histogram(counter):
    """
    Arguments
        counter: collections.Counter

    Returns a normalized collections.Counter
    """
    total = sum(counter.values(), 0.0)
    if total == 0:
        return counter
    
    for key in counter:
        counter[key] /= total

    return counter


def histogram_to_list(chorale_histogram, dataset_histogram):
    """
    Arguments
        chorale_histogram: Counter – this is the histogram that needs to be compared against the baseline
        dataset_histogram: Counter – this is the baseline histogram to compare chorale_histogram against

    Returns two lists of the same length, with each element being the count from the histograms in the order of dataset_histogram's keys.
    """
    ordered_keys = [x[0] for x in dataset_histogram.most_common()]
    ordered_dataset_histogram_vals = [x[1] for x in dataset_histogram.most_common()]
    ordered_chorale_histogram_vals = []
    chorale_histogram_extras = Counter()

    for okey in ordered_keys:  # make chorale_histogram the same order as dataset_histogram
        ordered_chorale_histogram_vals.append(chorale_histogram[okey])

    for key in chorale_histogram:  # get leftover vals from chorale_histogram in ascending order
        if key not in dataset_histogram:
            chorale_histogram_extras[key] += chorale_histogram[key]
    chorale_histogram_extras = [x[1] for x in chorale_histogram_extras.most_common()]
    chorale_histogram_extras.reverse()

    # add the leftovers
    ordered_dataset_histogram_vals.extend([0] * len(chorale_histogram_extras))
    ordered_chorale_histogram_vals.extend(chorale_histogram_extras)

    if len(dataset_histogram) < len(chorale_histogram):
        assert ordered_dataset_histogram_vals[-1] == 0
    assert len(ordered_dataset_histogram_vals) == len(ordered_chorale_histogram_vals)

    return ordered_chorale_histogram_vals, ordered_dataset_histogram_vals


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
            generation_scores.append(2-np.min([float(row[col]), 2]))

    # plot histograms
    plt.figure()
    bins = np.histogram(np.hstack((chorale_scores, generation_scores)), bins=30)[1]
    plt.hist(chorale_scores, label='Real chorales', alpha=0.5, bins=bins)
    plt.hist(generation_scores, label='Generated chorales', alpha=0.5, bins=bins)
    plt.xlabel(title)
    plt.ylabel('Frequency')
    # plt.title('Score distribution for real and generated chorales (histograms)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'combined_hist_{title}.png'))

    # plot boxplots
    plt.figure()
    boxplot_data = [chorale_scores, generation_scores]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data)
    ax.set_xticklabels(['Real chorales', 'Generated chorales'])
    plt.ylabel(title)
    # plt.title('Score distribution for real and generated chorales (boxplot)')
    plt.savefig(os.path.join(plot_dir, f'boxplots.png'))


