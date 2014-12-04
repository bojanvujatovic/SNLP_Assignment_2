# -*- coding: utf-8 -*-
"""main.py

Main file of the project containing main function. Defines the project 
workflow.

"""

from Classes.Sentences import *
from Features.example_features import *
from functools import partial
from classifier.naivebayes.naivebayes import *

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
    all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)
    
    #test_sentences = Paragraphs("Dataset/Test/").all_sentences()
    
    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    trigger_dict = train_sentences.get_trigger_dict()
    
    #for word in trigger_dict:
    #    print word
    
    print len(word_dict)
    print len(trigger_dict)
    
    phi = partial(whole_set_of_features, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))
    
    nb = NaiveBayes(phi, class_dict)
    
    trained_nb = nb.train(train_sentences.tokens())
    
if __name__ == "__main__":
    main()
