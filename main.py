# -*- coding: utf-8 -*-
"""main.py

Main file of the project containing main function. Defines the project 
workflow.

"""

from Classes.Sentences import *
from Features.example_features import *

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
    all_train_sentences = Paragraphs("Dataset/Train_single/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)
    
    #test_sentences = Paragraphs("Dataset/Test/").all_sentences()
    
    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    
    print class_dict
    
    word_class_template_feature(train_sentences.sentences[1].tokens[1], "None", None, word_dict, class_dict)

if __name__ == "__main__":
    main()