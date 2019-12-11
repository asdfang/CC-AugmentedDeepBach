"""
Call things from main()
"""

import sys
sys.path.insert(0, '../')

import music21
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pickle

from grader.histogram_helpers import *
from grader.compute_chorale_histograms import *
from scipy.stats import wasserstein_distance
    
# TODO: refactor histogram/distribution (don't forget about in the comments too)
'''
Saves a plot, and returns a score

ex: get_chorale_note_distribution_and_score('../generations/4/c0.mid', '../plots/bach_hist/genex_note_distribution.png',
                                histograms['major_note_histogram'], histograms['minor_note_histogram'],
                                major_note_order, minor_note_order)
'''
def get_chorale_note_distribution_and_score(chorale_filename, plot_filename,
                                                major_note_distribution, minor_note_distribution
                                                major_note_order, minor_note_order):
    """
    Arguments
        chorale_filename: String that holds path to a music21.stream.Stream
        plot_filename: String that holds where user wants the plot to be saved
        major_note_distribution: Counter holding note distribution of major keys
        minor_note_distribution: Counter holding note distribution of minor keys
        major_note_order: tuple holding order of notes to display on plot for major keys
        minor_note_order: tuple holding order of notes to display on plot for minor keys

    Saves a plot, and returns a score
    """
    chorale = music21.converter.parse(chorale_filename)
    key = gen.analyze('key')
    chorale_histogram = normalize_histogram(get_note_histogram(chorale, key))
    note_distribution = major_note_distribution if key.mode == 'major' else minor_note_distribution
    notes = major_note_order if key.mode == 'major' else minor_note_order

    y_pos = np.arange(len(notes))
    y_vals = chorale_histogram

    plt.bar(y_pos, y_vals, align='center')
    plt.xticks(y_pos, objects)
    plt.xlabel('Scale Degree')
    plt.ylabel('Proportion')
    plt.title('Generated Example Note Distribution')

    plt.savefig(plot_filename)

    return wasserstein_distance(*histogram_to_list(chorale_histogram, note_distribution))

# TODO: refactor histogram/distribution (don't forget about in the comments too)
'''
Example to plot major note distribution:
note_distribution_plot('../plots/bach_hist/major_note_histogram.png', 'Major Note Distribution', histograms['major_note_histogram'], major_note_order)
'''
def note_distribution_plot(plot_filename, plot_title, note_distribution, note_order):
    """
    Arguments
        plot_filename: String that holds where user wants the plot to be saved
        plot_title: what plot title should be
        note_distribution: Counter holding note distribution
        note_order: Tuple holding order o notes to display on plot
    """
    y_pos = np.arange(len(note_order))
    y_vals = [x[1] for x in note_distribution.most_common()]

    plt.bar(y_pos, y_vals, align='center')
    plt.xticks(y_pos, objects)
    plt.xlabel('Scale Degree')
    plt.ylabel('Proportion')

    # might want to modify this title
    plt.title('Note Distribution')
    plt.savefig(plot_filename)

    return

def main():
    histograms_file = 'grader/bach_histograms.txt'
    error_note_ratio_file = 'grader/error_note_ratio.txt'
    parallel_error_note_ratio_file = 'grader/parallel_error_note_ratio.txt'

    major_note_order = ('5', '1', '3', '2', '6', '4', '7', '4♯', '7♭', 'Rest', '1♯', '5♯', '3♭', '2♯', '6♭', '2♭')
    minor_note_order = ('5', '1', '3', '4', '2', '7', '6', '7♯', '6♯', '3♯', 'Rest', '4♯', '2♭', '5♭', '1♯', '1♭')

    with open(histograms_file, 'rb') as fin:
        histograms = pickle.load(fin)
    with open(error_note_ratio_file, 'rb') as fin:
        error_note_ratio = pickle.load(fin)
    with open(parallel_error_note_ratio_file, 'rb') as fin:
        parallel_error_note_ratio = pickle.load(fin)

    # # bug me if this doesn't work
    # get_chorale_note_distribution_and_score('../generations/4/c0.mid', '../plots/bach_hist/genex_note_distribution.png',
    #                             histograms['major_note_histogram'], histograms['minor_note_histogram'],
    #                             major_note_order, minor_note_order)
    # note_distribution_plot('../plots/bach_hist/major_note_histogram.png', 'Major Note Distribution', histograms['major_note_histogram'], major_note_order)

if __name__ == '__main__':
    main()