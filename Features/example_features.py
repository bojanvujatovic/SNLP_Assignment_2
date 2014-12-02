'''example_feature.py
'''

from scipy.sparse import *
from scipy import *

def word_template_feature(event_candidate_args, word_dict, token, event_candidate):
    
    N_words = len(word_dict)
    
    data = array([1])
    i = array([word_dict[token.word]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(N_words, 1), dtype=int8)

# Word belongs to a class                      
def word_class_template_feature(event_candidate_args, word_dict, class_dict, token, event_candidate):
    
    N_words = len(word_dict)
    N_classes = len(class_dict)

    data = array([1])
    i = array([word_dict[token.word] * N_classes + class_dict[event_candidate]])
    j = array([0]) 

    return csc_matrix((data, (i, j)), shape=(N_words*N_classes, 1), dtype=int8)

# Token has a capital letter
def capital_leter_feature(class_dict, token, event_candidate):
    
    N_classes = len(class_dict)
    
    if token.word[0].isupper(): 
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# Token is in the trigger dictionary
def token_in_trigger_dict_feature(class_dict, trigger_dict, token, event_candidate):
    
    N_classes = len(class_dict)
    
    if token.word in trigger_dict:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# Token has a number
def number_in_token_feature(class_dict, token, event_candidate):
    
    N_classes = len(class_dict)
        
    if any([char.isdigit() for char in token.word]):
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# Token is in a protein
def token_in_protein_feature(class_dict, token, event_candidate):
    
    N_classes = len(class_dict)
    
    if ("Protein" or "protein") in token.mentions:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# Token is after "-"
def token_is_after_dash_feature(class_dict, token, event_candidate):
    
    N_classes = len(class_dict)
    
    if token.word[0] == "-": 
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# POS tags combined with class; events are usually nouns, verbs or adjectives
def pos_class_feature(class_dict, token, event_candidate):
        
    pos_tags = ['NN','NNP','NNS','NNPS',
                'VB','VBD','VBG','VBN','VBP','VBZ',
                'JJ','JJR','JJS']
    # append the protein entry to the class list
    class_dict.update({"Protein":len(class_dict)+1})
    N_classes = len(class_dict)
    
    if token.pos in pos_tags:
        data = array([1])
    else:
        data = array([0])
    i = array([class_dict[event_candidate]])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(N_classes, 1), dtype=int8)

# Ngrams of characters
def character_ngram_feature(n, ngram_combinations, class_dict, token, event_candidate):
    
    N_classes = len(class_dict)
    N_ngrams = len(ngram_combinations)
    ngrams = [token.word[i:i+n] for i in range(len(token.word)-n+1)]
     
    data = array([1] * len(ngrams))
    i = array([index * N_classes + class_dict[event_candidate] 
              for index in [ngram_combinations[ngram] for ngram in ngrams]])
    j = array([0] * len(ngrams))
    
    print csc_matrix((data, (i, j)), shape=(N_ngrams*N_classes, 1), dtype=int8)
    return csc_matrix((data, (i, j)), shape=(N_ngrams*N_classes, 1), dtype=int8)
    
    