import music21
from collections import Counter

# TODO: TEST THIS!!!!!
# TODO: document
# TODO: convert final histogram to score

def find_voice_leading_errors(chorale):
    assert len(chorale.parts) == 4

    error_histogram = Counter()
    for top_idx in range(0, 3):
        for bottom_idx in range(top_idx+1, 4):
            top = chorale.parts[top_idx].notes
            bottom = chorale.parts[bottom_idx]

            # go backwards; leave first note out since current note i still needs a note to move from
            for i in range(len(top)-1, 0, -1):
                top_n1 = top[i-1]
                top_n2 = top[i]

                # make sure top voice's two notes are consecutive
                if top_n1.offset + top_n1.duration.quarterLength != top_n2.offset:
                    continue

                # make sure bottom voice also has a note that occurs exactly with the top voice's second note
                bottom_n2 = bottom.getElementsByOffset(offsetStart=top_n2.offset, classList=music21.note.Note)
                if len(bottom_n2) == 0:
                    continue
                assert len(bottom_n2) == 1
                bottom_n2 = bottom_n2[0]

                # make sure bottom voice's two notes are also consecutive
                bottom_n1 = bottom.getElementBeforeOffset(offset=bottom_n2.offset, classList=music21.note.Note)
                if (bottom_n1 is None) or (bottom_n1.offset + bottom_n1.duration.quarterLength != bottom_n2.offset):
                    continue

                # find voice leading mistakes!
                vlq = music21.voiceLeading.VoiceLeadingQuartet(top_n1, top_n2, bottom_n1, bottom_n2)

                # TODO: this function doesn't return anything yet
                if vlq.parallelUnisonOrOctave():
                    error_histogram['Prl-8ve'] += 1
                if vlq.parallelFifth():
                    error_histogram['Prl-5th'] += 1
                if vlq.hiddenOctave():
                    error_histogram['H-8ve'] += 1
                if vlq.hiddenFifth():
                    error_histogram['H-5th'] += 1
                if vlq.voiceOverlap():
                    error_histogram['Overlap'] += 1
    
    return error_histogram

# helper for find_voice_crossing_and_spacing_errors()
def notes_sound_simultaneously(n1, n2):
    assert n1.isNote
    assert n2.isNote
    if n1.offset == n2.offset:
        return True
    if n1.offset < n2.offset:
        return n1.offset+n1.duration.quarterLength > n2.offset
    else:
        return n2.offset+n2.duration.quarterLength > n1.offset


def find_voice_crossing_and_spacing_errors(chorale):
    assert len(chorale.parts) == 4

    error_histogram = Counter()
    for top_idx in range(0, 3):
        for bottom_idx in range(top_idx+1, 4):
            top = chorale.parts[top_idx].notes
            bottom = chorale.parts[bottom_idx].notes
            ti = bi = 0
            while ti < len(top) and bi < len(bottom):
                n1 = top[ti]
                n2 = bottom[bi]

                if notes_sound_simultaneously(n1, n2):
                    dummy = music21.note.Note('C4')
                    vlq = music21.voiceLeading.VoiceLeadingQuartet(n1, dummy, n2, dummy)
                    if vlq.voiceCrossing():
                        error_histogram['Crossing'] += 1

                    if (top_idx == 0 and bottom_idx == 1) or (top_idx == 1 and bottom_idx == 2):
                         if abs(music21.interval.Interval(noteStart=n1, noteEnd=n2).semitones) > 12:
                            error_histogram['Spacing'] += 1

                # increment whichever ends first
                if n1.offset+n1.quarterLength < n2.offset+n2.quarterLength:
                    ti += 1
                else:
                    bi += 1

    return error_histogram

voice_ranges = [(60, 81), (53, 74), (48, 69), (36, 64)]
# impo: ranges calculate from actual Bach chorales, instead of what textbooks suggest
def find_voice_range_errors(chorale):
    assert len(chorale.parts) == 4

    error_histogram = Counter()
    for part_idx in range(0, 4):
        for note in chorale.parts[part_idx].notes:
            curr_range = voice_ranges[part_idx]
            midi = note.pitch.midi
            if midi < curr_range[0] or midi > curr_range[1]:
                error_histogram['Range'] += 1

    return error_histogram

def find_part_writing_errors(chorale):
    possible_errors = ['Prl-8ve', 'Prl-5th', 'H-8ve', 'H-5th', 'Overlap', 'Crossing', 'Spacing', 'Range']

    # initialize counts to 0
    error_histogram = Counter()
    error_histogram.update({error:0 for error in possible_errors})

    error_histogram += find_voice_leading_errors(chorale) + find_voice_crossing_and_spacing_errors(chorale) + find_voice_range_errors(chorale)
    return error_histogram


# testing
# bad_chorale = music21.converter.parse('errors.mid')
# eh = find_part_writing_errors(bad_chorale)