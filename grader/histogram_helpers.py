from collections import Counter


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

# TODO: incorporate this into grader.py, chorale_datasets.py
# import music21
# from tqdm import tqdm
# all_directed_ih = Counter()
# all_undirected_ih = Counter()
# for chorale in tqdm(music21.corpus.chorales.Iterator(1,371)):
#     if len(chorale.parts) == 4:
#         r1, r2 = get_interval_histogram(chorale)
#         all_directed_ih += r1
#         all_undirected_ih += r2
def get_interval_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns two interval histograms, one directed and one undirected,
    as collections.Counter objects for input chorale
    """
    directed_ih = Counter()
    undirected_ih = Counter()
    
    assert len(chorale.parts) == 4
    for part in chorale.parts:
        intervals = part.melodicIntervals()[1:] # all but first meaningless result
        for interval in intervals:
            directed_ih[interval.directedName] += 1
            undirected_ih[interval.name] += 1

    return directed_ih, undirected_ih


def normalize_histogram(counter):
    """
    Arguments
        counter: collections.Counter

    Returns a normalized collections.Counter
    """
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total

    return counter

