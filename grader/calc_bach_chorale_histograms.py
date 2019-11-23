import music21
import collections
from tqdm import tqdm


# from DeepBach
def is_valid(chorale):
    if not len(chorale.parts) == 4:
        return False
    return True


def chorale_note_histogram(chorale, key):
    nh = collections.Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        if note_or_rest.isNote:
            sd = key.getScaleDegreeAndAccidentalFromPitch(note_or_rest.pitch)
            nh[sd] += 1
        else:
            nh['Rest'] += 1
    return nh


def chorale_rhythm_histogram(chorale):
    rh = collections.Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        rh[note_or_rest.duration.quarterLength] += 1
    return rh


def chorale_interval_histogram(chorale):
    ih = collections.Counter()
    chorale.parts[0].melodicIntervals().show('txt')
    chorale.show()


major_nh = collections.Counter()
minor_nh = collections.Counter()
all_rh = collections.Counter()
all_ih = collections.Counter()
for fname in tqdm(music21.corpus.chorales.Iterator(3, 3, returnType='filename')):
    chorale = music21.corpus.parse(fname)
    if is_valid(chorale):
        key = chorale.analyze('key')

        # note histograms
        chorale_nh = chorale_note_histogram(chorale, key)
        if key.mode == 'major':
            major_nh += chorale_nh
        else:
            minor_nh += chorale_nh

        # rhythm histogram
        all_rh += chorale_rhythm_histogram(chorale)

        # interval histogram
        all_ih += chorale_interval_histogram(chorale)

print(all_ih)

# print('major:')
# print(major_nh)
# print('minor:')
# print(minor_nh)
# print('rhythm:')
# print(all_rh)
