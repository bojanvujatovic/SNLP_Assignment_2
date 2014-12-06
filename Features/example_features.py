'''example_feature.py
'''

from scipy.sparse import csr_matrix
from scipy import array


def word_template_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                          ngram_combinations, token, event_candidate):
    try:
        i = [word_dict[token.word]]
    except:
        i = [word_dict['<<UNK>>']]

    return i, len(word_dict)


# Word belongs to a class
def word_class_template_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                                ngram_combinations, token, event_candidate):
    n_words = len(word_dict)
    n_classes = len(class_dict)

    try:
        i = [word_dict[token.word] * n_classes + class_dict[event_candidate]]
    except:
        i = [word_dict['<<UNK>>'] * n_classes + class_dict[event_candidate]]

    return i, n_words * n_classes


# Token has a capital letter
def capital_letter_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                           ngram_combinations, token, event_candidate):
    if token.word[0].isupper():
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# Token is in the trigger dictionary
def token_in_trigger_dict_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                                  ngram_combinations, token, event_candidate):
    if token.word in trigger_dict:
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# Token has a number
def number_in_token_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                            ngram_combinations, token, event_candidate):
    if any([char.isdigit() for char in token.word]):
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# Token is in a protein
def token_in_protein_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                             ngram_combinations, token, event_candidate):
    if ("Protein" or "protein") in token.mentions:
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# Token is after "-"
def token_is_after_dash_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                                ngram_combinations, token, event_candidate):
    if token.word[0] == "-":
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# POS tags combined with class; events are usually nouns, verbs or adjectives
def pos_class_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                      ngram_combinations, token, event_candidate):
    pos_tags = ['NN', 'NNP', 'NNS', 'NNPS',
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS']

    if token.pos in pos_tags:
        i = [class_dict[event_candidate]]
    else:
        i = []

    return i, len(class_dict)


# Ngrams of characters
def character_ngram_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                            ngram_combinations, token, event_candidate):
    n_classes = len(class_dict)
    n_ngrams = len(char_ngram_combinations)
    ngrams = [token.word[i:i + n] for i in range(len(token.word) - n + 1)]

    ngram_indices = []
    for ngram in ngrams:
        try:
            ngram_indices.append(char_ngram_combinations[ngram])
        except:
            ngram_indices.append(char_ngram_combinations['<<UNK>>'])

    i = [index * n_classes + class_dict[event_candidate] for index in ngram_indices]

    return i, n_ngrams * n_classes


def word_stem_feature(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                      ngram_combinations, token, event_candidate):
    try:
        i = [stem_dict[token.stem]]
    except:
        i = [stem_dict['<<UNK>>']]

    return i, len(stem_dict)


def ngram_features(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                   ngram_combinations, token, event_candidate):
    n_classes = len(class_dict)
    n_ngrams = len(ngram_combinations)

    # construct the token ngram
    if token.index == 0:
        ngram = ['<<START>>', str(token.word)]
    elif (token.tokens_in_sentence[-1].word == token.word) and \
            (len(token.tokens_in_sentence) - 1 == token.index):
        ngram = [str(token.word), '<<END>>']
    else:
        ngram = [str(token.tokens_in_sentence[token.index - 1].word), str(token.word)]

    # check if the ngram is in the ngram list
    try:
        i = [ngram_combinations[ngram] * n_classes + class_dict[event_candidate]]
    except:
        i = [ngram_combinations['<<UNK>>'] * n_classes + class_dict[event_candidate]]

    return i, n_ngrams * n_classes


def set_of_features(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations, ngram_combinations,
                    feature_strings, token, event_candidate):
    matrix_indices = []
    matrix_length = 0
    for feature_string in feature_strings:
        res = globals()[feature_string](stem_dict, word_dict, class_dict, trigger_dict, n,
                                        char_ngram_combinations, ngram_combinations, token, event_candidate)
        matrix_indices += [x + matrix_length for x in res[0]]
        matrix_length += res[1]

    v_length = len(matrix_indices)
    data = array([1] * v_length)
    i = array(matrix_indices)
    j = array([0] * v_length)
    return csr_matrix((data, (i, j)), shape=(matrix_length, 1))


def set_of_features_structured(stem_dict, word_dict, class_dict, trigger_dict, n, char_ngram_combinations,
                               ngram_combinations, feature_strings, token, event_candidate):
    matrix_indices = []
    matrix_length = 0

    res = word_stem_feature(stem_dict, None, None, None, None, None, None, token[0], None)
    matrix_indices += [x + matrix_length for x in res[0]]
    matrix_length += res[1]

    for feature_string in feature_strings:
        res = globals()[feature_string](stem_dict, word_dict, class_dict, trigger_dict, n,
                                        char_ngram_combinations, ngram_combinations, token[1], event_candidate)
        matrix_indices += [x + matrix_length for x in res[0]]
        matrix_length += res[1]

    v_length = len(matrix_indices)
    data = array([1] * v_length)
    i = array(matrix_indices)
    j = array([0] * v_length)
    return csr_matrix((data, (i, j)), shape=(matrix_length, 1))
