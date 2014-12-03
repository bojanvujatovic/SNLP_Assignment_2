import random
import time


def subsample_none(tokens, fraction, seed=time.clock()):
    random.seed(seed)
    return [t for t in tokens if t.event_candidate != 'None' or random.random() < fraction]


