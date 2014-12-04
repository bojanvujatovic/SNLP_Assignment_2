# -*- coding: utf-8 -*-
"""main.py

Main file of the project containing main function. Defines the project 
workflow.

"""

from Classes.Sentences import *
from Features.example_features import *
from functools import partial
from classifier.naivebayes.naivebayes import *
from utils.Utils import *

def main():
    """Main function of the project.
    # QUESTO E IL MIO COMMENTO
    # Esto es un comentario en español
    # Isto é un comentario en galego
    Defines the project structure calling the following functions:
    1. 
    2. 
    3.
    """
    all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)
    
    #test_sentences = Paragraphs("Dataset/Test/").all_sentences()
    
    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    trigger_dict = train_sentences.get_trigger_dict()
    stem_dict = get_stem_dict(train_sentences.tokens())
    
    #for word in trigger_dict:
    #    print word
    
    nb = NaiveBayes(stem_dict, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))
    
    feature_strings = ["capital_letter_class_feature", "class_feature", "word_class_feature", "token_in_trigger_dict_class_feature", "number_in_token_class_feature", "word_stem_class_feature"]
    trained_nb = nb.train(train_sentences.tokens(), feature_strings)
    
    for s in hand_out_sentences.sentences:
        for t in s.tokens:
            print t.event_candidate, trained_nb.predict(t)
    
if __name__ == "__main__":
    main()
