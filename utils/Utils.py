import random
import time


def subsample_none(tokens, fraction, seed=time.clock()):
    random.seed(seed)
    return [t for t in tokens if t.event_candidate != 'None' or random.random() < fraction]


def subsample_label(tokens, label, fraction, seed=time.clock()):
    random.seed(seed)
    return [t for t in tokens if t.event_candidate != label or random.random() < fraction]


def get_word_dict(tokens):
    word_dict = dict(map(lambda p: (p[1], p[0]), enumerate(set([t.word for t in tokens]))))
    word_dict['<<UNK>>'] = len(word_dict)
    return word_dict


def get_class_dict(tokens):
    return dict(map(lambda p: (p[1], p[0]), enumerate(set([t.event_candidate for t in tokens]))))


def get_stem_dict(tokens):
    stem_dict = dict(map(lambda p: (p[1], p[0]), enumerate(set([t.stem for t in tokens]))))
    stem_dict['<<UNK>>'] = len(stem_dict)
    return stem_dict


def get_char_ngram_dict(tokens, n):
    ngram_dict = dict(map(lambda p: (p[1], p[0]),
                          enumerate(set([t.word[i:i + n] for t in tokens for i in range(len(t.word) - n + 1)]))))
    ngram_dict['<<UNK>>'] = len(ngram_dict)
    return ngram_dict


def get_ngram_dict(tokens, n):
    # l0 = [str(map(lambda t: t.word, tokens[i: i + n])) for i in range(len(tokens) - n + 1)]
    # l = set(l0)
    # ngram_dict = dict(map(lambda p: (p[1], p[0]), enumerate(l)))
    # ngram_dict['<<UNK>>'] = len(ngram_dict)
    # return ngram_dict
    ngrams = set([])
    current_ngram = []
    for token in tokens:
        if token.index == 0:
            current_ngram = (['<<START>>'] * (n - 1)) + [token.word]
        elif token.index == len(token.tokens_in_sentence) - 1:
            current_ngram = current_ngram[1:] + [token.word]
            for _ in range(n, 1, -1):
                ngrams.add(str(current_ngram))
                current_ngram = current_ngram[1:] + ['<<END>>']
        else:
            current_ngram = current_ngram[1:] + [token.word]
        ngrams.add(str(current_ngram))
    ngrams.add('<<UNK>>')
    return dict(map(lambda p: (p[0], p[1]), enumerate(ngrams)))


def get_trigger_dict(tokens):
    return dict(map(lambda p: (p[1], p[0]),
                    enumerate(
                        set([trig for t in tokens if t.event_candidate != 'None' for trig in t.word.split('-')]))))


def get_arg_word_dict(tokens):
    arg_words = set([t[0]['word'] for token in tokens if token.event_candidate != 'None'
                     for t in token.event_candidate_args if t[1] != 'None'])
    arg_words.add('<<UNK>>')
    return dict(map(lambda p: (p[1], p[0]), enumerate(arg_words)))