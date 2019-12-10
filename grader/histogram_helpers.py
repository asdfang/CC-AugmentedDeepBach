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


def histogram_to_list(h1, h2):
    """
    Arguments
      h1: Counter – this is the baseline histogram to compare h2 against
      h2: Counter – this is the histogram that needs to be compared against the baseline

    Returns two lists of the same length, with each element being the count from the histograms in the order of h1's keys.
    """
    ordered_keys = [x[0] for x in h1.most_common()]
    ordered_h1_vals = [x[1] for x in h1.most_common()]
    ordered_h2_vals = []
    h2_extras = Counter()

    for okey in ordered_keys:  # make h2 the same order as h1
        ordered_h2_vals.append(h2[okey])

    for key in h2:  # get leftover vals from h2 in ascending order
        if key not in h1:
            h2_extras[key] += h2[key]
    h2_extras = [x[1] for x in h2_extras.most_common()]
    h2_extras.reverse()

    # add the leftovers
    ordered_h1_vals.extend([0] * len(h2_extras))
    ordered_h2_vals.extend(h2_extras)

    if len(h1) < len(h2):
        assert ordered_h1_vals[-1] == 0
    assert len(ordered_h1_vals) == len(ordered_h2_vals)

    return ordered_h1_vals, ordered_h2_vals


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
            chorale_scores.append(float(row[col]))

    with open(generation_file, 'r') as generation_file:
        reader = csv.reader(generation_file)
        for i, row in enumerate(reader):
            if i == 0:
                assert row[col] == title
                continue
            generation_scores.append(float(row[col]))

    plt.figure()
    bins = np.histogram(np.hstack((chorale_scores, generation_scores)), bins=20)[1]
    plt.hist(chorale_scores, label='real chorales', alpha=0.5, bins=bins)
    plt.hist(generation_scores, label='generated chorales', alpha=0.5, bins=bins)
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'combined_hist_{title}.png'))
