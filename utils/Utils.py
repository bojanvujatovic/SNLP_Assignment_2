import random
import time


def subsample_none(tokens, fraction, seed=time.clock()):
    random.seed(seed)
    return [t for t in tokens if t.event_candidate != 'None' or random.random() < fraction]

def subsample_label(tokens, label, fraction, seed=time.clock()):
    random.seed(seed)
    return [t for t in tokens if t.event_candidate != label or random.random() < fraction]


def get_word_dict(tokens):
    return dict(map(lambda p: (p[1], p[0]), enumerate(set([t.word for t in tokens]))))


def get_class_dict(tokens):
    return dict(map(lambda p: (p[1], p[0]), enumerate(set([t.event_candidate for t in tokens]))))


def get_stem_dict(tokens):
    return dict(map(lambda p: (p[1], p[0]), enumerate(set([t.stem for t in tokens]))))
