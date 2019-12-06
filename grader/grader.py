"""
score a chorale compared to a dataset of ground-truth chorales
"""

from grader.get_feature_scores import *


score_methods_dict = {'error': get_error_score,
                      'note': get_note_score,
                      'rhythm': get_rhythm_score,
                      'undirected_interval': get_undirected_interval_score,
                      'directed_interval': get_directed_interval_score}


def score_chorale(chorale, dataset, features=None):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object
        scores: a dict of weights for each score component

    return overall score and component scores
    """

    assert dataset.histograms is not None

    if not features:
        features = {feature: 1 for feature in score_methods_dict}

    scores = []
    score = 0

    for f in features:
        score = score_methods_dict[f](chorale, dataset)
        weight = features[f]
        scores.append(score)
        score += weight*score

    return score, scores


