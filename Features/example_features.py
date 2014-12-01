'''example_feature.py
'''

from scipy.sparse import *
from scipy import *

def word_template_feature(token, event_candidate, event_candidate_args, word_dict):
    
    N_words = len(word_dict)
    
    data = array([1])
    i = array([word_dict[token.word]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(N_words, 1), dtype=int8)
    
def word_class_template_feature(token, event_candidate, event_candidate_args, word_dict, class_dict):
    
    N_words = len(word_dict)
    N_classes = len(class_dict)

    data = array([1])
    i = array([word_dict[token.word] * N_classes + class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(N_words*N_classes, 1), dtype=int8)