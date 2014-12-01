# -*- coding: utf-8 -*-
"""main.py

Main file of the project containing main function. Defines the project 
workflow.

"""

from Classes.Sentences import *

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
    
    test_sentences = Paragraphs("Dataset/Test/").all_sentences()
    
    print len(train_sentences.sentences)
    print len(hand_out_sentences.sentences)
    print len(test_sentences.sentences)

if __name__ == "__main__":
    main()