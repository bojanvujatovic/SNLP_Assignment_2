'''example_feature.py
'''

from scipy.sparse import *
from scipy import *


def word_template_feature(word_dict, token):
    n_words = len(word_dict)

    data = array([1])
    i = array([word_dict[token.word]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_words, 1), dtype=int8)


# Word belongs to a class                      
def word_class_template_feature(word_dict, class_dict, token, event_candidate):
    n_words = len(word_dict)
    n_classes = len(class_dict)

    data = array([1])
    i = array([word_dict[token.word] * n_classes + class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_words * n_classes, 1), dtype=int8)

# Token has a capital letter
def capital_leter_feature(class_dict, token, event_candidate):
    n_classes = len(class_dict)

    if token.word[0].isupper():
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# Token is in the trigger dictionary
def token_in_trigger_dict_feature(class_dict, trigger_dict, token, event_candidate):
    n_classes = len(class_dict)

    if token.word in trigger_dict:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# Token has a number
def number_in_token_feature(class_dict, token, event_candidate):
    n_classes = len(class_dict)

    if any([char.isdigit() for char in token.word]):
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# Token is in a protein
def token_in_protein_feature(class_dict, token, event_candidate):
    n_classes = len(class_dict)

    if ("Protein" or "protein") in token.mentions:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# Token is after "-"
def token_is_after_dash_feature(class_dict, token, event_candidate):
    n_classes = len(class_dict)

    if token.word[0] == "-":
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# POS tags combined with class; events are usually nouns, verbs or adjectives
def pos_class_feature(class_dict, token, event_candidate):
    pos_tags = ['NN', 'NNP', 'NNS', 'NNPS',
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS']
    # append the protein entry to the class list
    class_dict.update({"Protein": len(class_dict) + 1})
    n_classes = len(class_dict)

    if token.pos in pos_tags:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(n_classes, 1), dtype=int8)


# Ngrams of characters
def character_ngram_feature(n, ngram_combinations, class_dict, token, event_candidate):
    n_classes = len(class_dict)
    n_grams = len(ngram_combinations)
    ngrams = [token.word[i:i + n] for i in range(len(token.word) - n + 1)]

    data = array([1] * len(ngrams))
    i = array([index * n_classes + class_dict[event_candidate]
               for index in [ngram_combinations[ngram] for ngram in ngrams]])
    j = array([0] * len(ngrams))

    # print csc_matrix((data, (i, j)), shape=(n_grams * n_classes, 1), dtype=int8)
    return csc_matrix((data, (i, j)), shape=(n_grams * n_classes, 1), dtype=int8)


def whole_set_of_features(word_dict, class_dict, trigger_dict, n, ngram_combinations, token, event_candidate):
    # return vstack([word_template_feature(word_dict, token),
    #                word_class_template_feature(word_dict, class_dict, token, event_candidate),
    #                capital_leter_feature(class_dict, token, event_candidate),
    #                token_in_trigger_dict_feature(class_dict, trigger_dict, token, event_candidate),
    #                number_in_token_feature(class_dict, token, event_candidate),
    #                token_in_protein_feature(class_dict, token, event_candidate),
    #                token_is_after_dash_feature(class_dict, token, event_candidate),
    #                pos_class_feature(class_dict, token, event_candidate),
    #                character_ngram_feature(n, ngram_combinations, class_dict, token, event_candidate)])
    return word_template_feature(word_dict, token)
