'''example_feature.py
'''

import Classes

def example_feature(token, event_candidate, event_candidate_args, compare_word):
    
    if token.word == compare_word:
        return 1
    else:
        return 0