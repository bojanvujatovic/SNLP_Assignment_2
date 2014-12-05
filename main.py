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
from classifier.ErrorAnalysis import *

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
    
    # READ DATA
    
    used_fraction = 1
    train_fraction = 0.8
    none_fraction = 0.1
    
    all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)
    
    (used_sentences, _) = all_train_sentences.split_randomly(used_fraction)
    (train_sentences, test_sentences) = used_sentences.split_randomly(train_fraction)

    all_train_tokens = train_sentences.tokens()
    subsampled_tokens = subsample_none(all_train_tokens, none_fraction)
    
    class_dict = get_class_dict(subsampled_tokens)
    stem_dict = get_stem_dict(subsampled_tokens)
    word_dict = get_word_dict(subsampled_tokens)
    trigger_dict = get_trigger_dict(subsampled_tokens)
    
    
    # TRAINING
    
    nb = NaiveBayes(stem_dict, word_dict, class_dict, trigger_dict)
    
    feature_strings = ["capital_letter_class_feature", 
                       "class_feature", 
                       "word_class_feature", 
                       "token_in_trigger_dict_class_feature", 
                       "number_in_token_class_feature", 
                       "word_stem_class_feature",
                       "pos_class_feature",
                       "token_is_after_dash_feature"]
    trained_nb = nb.train(subsampled_tokens, feature_strings)
    
    
    # ERROR ANALYSIS
    
    all_test_tokens = test_sentences.tokens()
    
    predictions = trained_nb.predict_all(all_test_tokens)
    
    true_labels = []
    for token in all_test_tokens:
        true_labels.append(token.event_candidate)

    test_keys = class_dict.keys()
    test_keys.pop(0)
    for label in test_keys:
        print 'Analyzing label: ', label
        precision_recall_f1(true_labels, predictions, label)

    from sklearn.metrics import confusion_matrix

    y_test = map(lambda t: t.event_candidate, all_test_tokens)
    y_pred = predictions

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # Show confusion matrix in a separate window
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    
    
if __name__ == "__main__":
    main()
