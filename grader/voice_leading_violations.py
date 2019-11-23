import music21


# will only find parallel octaves that happen truly consecutively
def find_voice_leading_errors(chorale):
    top = chorale.parts[0].notes
    bottom = chorale.parts[1]

    chorale.show('txt')

    # go backwards; leave first note out since current note i still needs a note to move from
    for i in range(len(top) - 1, 0, -1):
        # print('@@@@@@@@@@')
        top_n1 = top[i - 1]
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
        # print(bottom_n2)
        # print(bottom_n2.offset)

        # make sure bottom voice's two notes are also consecutive
        bottom_n1 = bottom.getElementBeforeOffset(offset=bottom_n2.offset, classList=music21.note.Note)
        if (bottom_n1 is None) or (bottom_n1.offset + bottom_n1.duration.quarterLength != bottom_n2.offset):
            continue

        # print(top_n1.nameWithOctave)
        # print(top_n2.nameWithOctave)
        # print(bottom_n1.nameWithOctave)
        # print(bottom_n2.nameWithOctave)

        # find voice leading mistakes!
        vlq = music21.voiceLeading.VoiceLeadingQuartet(top_n1, top_n2, bottom_n1, bottom_n2)

        # TODO: this function doesn't return anything yet
        vlq.parallelUnisonOrOctave()
        vlq.parallelFifth()
        vlq.hiddenOctave()
        vlq.hiddenFifth()
        vlq.voiceOverlap()

# testing
# p8 = music21.converter.parse('P8_octavedown.mid')
# find_voice_leading_errors(p8)
