from collections import Counter
import matplotlib.pyplot as plt


def get_note_histogram(chorale, key):
    nh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        if note_or_rest.isNote:
            sd = key.getScaleDegreeAndAccidentalFromPitch(note_or_rest.pitch)
            nh[sd] += 1
        else:
            nh['Rest'] += 1
    return nh


def get_rhythm_histogram(chorale):
    rh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        rh[note_or_rest.duration.quarterLength] += 1
    return rh


def get_interval_histogram(chorale):
    ih = Counter()
    chorale.parts[0].melodicIntervals().show('txt')
    chorale.show()


def normalize_histogram(counter):
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total

    return counter


def plot_distributions(dict):
    """
    Arguments
        dict: dictionary of score list

    plots many distributions on one graph, to visualize relationship between distributions
    """
    for key in dict:
        plt.hist(dict[key], label=key)

    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/score_distribution.png')

    with open('data/score_dict.txt', 'w') as fo:
        print(dict, file=fo)
