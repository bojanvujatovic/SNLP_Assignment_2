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

# Token has a capital letter
def capital_leter_feature(token):
    
    if token.word[0].isupper(): 
        data = array([1])
    else:
        data = array([0])
    i = array([0])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8)

# Token is in the trigger dictionary
def token_in_trigger_dict_feature(token, trigger_dict):
    
    if token.word in trigger_dict:
        data = array([1])
    else:
        data = array([0])
    i = array([0])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8)

# Token has a number
def number_in_token_feature(token):
        
    if any([char.isdigit() for char in token.word]):
        print "hi"
        data = array([1])
    else:
        data = array([0])
    i = array([0])
    j = array([0])
    print csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8).todense()
    return csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8)

# Token is in a protein
def token_in_protein_feature(token):
    
    if ("Protein" or "protein") in token.mentions:
        data = array([1])
    else:
        data = array([0])
    i = array([0])
    j = array([0])
    
    print csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8).todense()
    return csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8)

# Token is after "-"
def token_is_after_dash_feature(token):
    
    if token.word[0] == "-": 
        data = array([1])
    else:
        data = array([0])
    i = array([0])
    j = array([0])
    
    return csc_matrix((data, (i, j)), shape=(1, 1), dtype=int8)

# POS tags combined with class; events are usually nouns, verbs or adjectives
def pos_class_feature(token, event_candidate, word_dict, class_dict):
    
    pos_tags = ['NN','NNP','NNS','NNPS',
                     'VB','VBD','VBG','VBN','VBP','VBZ',
                     'JJ','JJR','JJS']
    N_classes = len(class_dict)
    N_tags = len(pos_tags)
    # append the protein entry to the class list
    class_dict.expand({"Protein":len(class_dict)+1})
    
    if token.pos in pos_tags:
        data = array([1])
    else:
        data = array([0])
    i = array([N_classes*N_tags + class_dict[event_candidate]])
    j = array([0])

    return csc_matrix((data, (i, j)), shape=(N_classes*N_tags, 1), dtype=int8)

# Bigrams of characters
def character_bigram_feature():
    return
    
    