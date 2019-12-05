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


def get_ordered_histograms(h1, h2):
  """
  Arguments
    h1: Counter – this is the baseline histogram to compare h2 against
    h2: Counter – this is the histogram that needs to be compared against the baseline

    Returns a tuple of two lists, both of the same length, with each element being the count from the histograms in the order of h1's keys.
  """
  ordered_keys = [x[0] for x in h1.most_common()]
  ordered_h1_vals = [x[1] for x in h1.most_common()]
  ordered_h2_vals = []
  h2_extras = Counter()

  for okey in ordered_keys: # make h2 the same order as h1
      ordered_h2_vals.append(h2[okey])
  
  for key in h2: # get leftover vals from h2 in ascending order
    if key not in h1:
      h2_extras[key] += h2[key]
  h2_extras = [x[1] for x in h2_extras.most_common()]
  h2_extras.reverse()

  # add the leftovers
  ordered_h1_vals.extend([0]*len(h2_extras))
  ordered_h2_vals.extend(h2_extras)

  if len(h1) < len(h2):
    assert ordered_h1_vals[-1] == 0
  assert len(ordered_h1_vals) == len(ordered_h2_vals)

  return ordered_h1_vals, ordered_h2_vals