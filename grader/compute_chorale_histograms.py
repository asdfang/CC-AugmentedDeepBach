"""
functions for computing the distribution for a given chorale
"""

from collections import Counter
from grader.voice_leading_helpers import *
import music21


def get_note_histogram(chorale, key):
    """
    Arguments
        chorale: a music21 Stream object
        key: music21.key.Key

    Returns a note histogram as a collections.Counter object for input chorale
        Counter key: (scale degree, accidental) or 'Rest'
        Counter value: count
    """
    nh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        if note_or_rest.isNote:
            sd = key.getScaleDegreeAndAccidentalFromPitch(note_or_rest.pitch)
            nh[sd] += 1
        else:
            nh['Rest'] += 1

    return nh


def get_rhythm_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns a rhythm histogram as a collections.Counter object for input chorale
        Counter key: float or 'Rest', notating a rhythm in terms of a quarter-note's length
            (i.e. 1.0: quarter-note, 0.5: eighth note, 2.0: half note, etc.)
        Counter value: count
    """
    rh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        rh[note_or_rest.duration.quarterLength] += 1

    return rh


def get_harmonic_quality_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns a harmonic quality (i.e. ignores root) histogram as calculated by music21.harmony.chordSymbolFigureFromChord()
    """

    hqh = Counter()       # harmony qualities
    
    chfy = chorale.chordify()
    for c in chfy.flat.getElementsByClass(music21.chord.Chord):
        csf = music21.harmony.chordSymbolFigureFromChord(c, True)
        if csf[0] == 'Chord Symbol Cannot Be Identified':
            hqh['unidentifiable'] += 1
        else:
            hqh[csf[1]] += 1

    return hqh


def get_interval_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns two interval histograms, one directed and one undirected,
    as collections.Counter objects for input chorale
    """
    directed_ih = Counter()
    undirected_ih = Counter()

    for part in chorale.parts:
        intervals = part.melodicIntervals()[1:]  # all but first meaningless result
        for interval in intervals:
            directed_ih[interval.directedName] += 1
            undirected_ih[interval.name] += 1

    return directed_ih, undirected_ih


def get_SATB_interval_histograms(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns two lists of interval histograms, one directed and one undirected,
    with each list containing one histogram per voice in the order of Soprano, Alto, Tenor, Bass.
    as collections.Counter objects for input chorale
    """
    assert len(chorale.parts) == 4

    directed_ihs = []
    S_directed_ih = Counter()
    A_directed_ih = Counter()
    T_directed_ih = Counter()
    B_directed_ih = Counter()
    directed_ihs.append(S_directed_ih)
    directed_ihs.append(A_directed_ih)
    directed_ihs.append(T_directed_ih)
    directed_ihs.append(B_directed_ih)

    undirected_ihs = []
    S_undirected_ih = Counter()
    A_undirected_ih = Counter()
    T_undirected_ih = Counter()
    B_undirected_ih = Counter()
    undirected_ihs.append(S_undirected_ih)
    undirected_ihs.append(A_undirected_ih)
    undirected_ihs.append(T_undirected_ih)
    undirected_ihs.append(B_undirected_ih)

    for i in range(0, 4):
        intervals = chorale.parts[i].melodicIntervals()[1:]
        for interval in intervals:
            directed_ihs[i][interval.directedName] += 1
            undirected_ihs[i][interval.name] += 1

    return directed_ihs, undirected_ihs
    

def get_error_histogram(chorale, voice_ranges):
    """
    Arguments
        chorale: a music21 Stream object; must have 4 parts once chorale gets passed into its helper functions

    Returns a count of voice leading errors founding in keys
        Counter
    """
    possible_errors = ['H-8ve', 'H-5th', 'Overlap', 'Crossing', 'Spacing', 'Range']

    # initialize counts to 0
    error_histogram = Counter()
    error_histogram += find_voice_leading_errors(chorale) + find_voice_crossing_and_spacing_errors(
        chorale) + find_voice_range_errors(chorale, voice_ranges)
    error_histogram.update({error: 0 for error in possible_errors})  # doesn't over-write

    return error_histogram

def get_parallel_error_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object; must have 4 parts once chorale gets passed into its helper functions

    Returns a count of voice leading errors founding in keys
        Counter
    """
    possible_errors = ['Prl-8ve', 'Prl-5th']

    # initialize counts to 0
    error_histogram = Counter()
    error_histogram += find_parallel_8ve_5th_errors(chorale)
    error_histogram.update({error: 0 for error in possible_errors})  # doesn't over-write

    return error_histogram