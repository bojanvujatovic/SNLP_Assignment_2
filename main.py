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
import matplotlib.pyplot as plt

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
    train_fraction = 0.3
    none_args_fraction = 0.01
    
    all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()
    
    (used_sentences, _) = all_train_sentences.split_randomly(used_fraction)
    (train_sentences, test_sentences) = used_sentences.split_randomly(train_fraction)

    all_train_tokens = train_sentences.tokens()
    all_test_tokens = test_sentences.tokens()
    subsampled_tokens = subsample_none_args(all_train_tokens, none_args_fraction)
    class_dict_events_args = get_class_args_dict(subsampled_tokens)
    
    class_dict_events = get_class_dict(subsampled_tokens)
    stem_dict = get_stem_dict_args(subsampled_tokens)
    word_dict = get_word_dict_args(subsampled_tokens)
    trigger_dict = get_trigger_dict(subsampled_tokens)
    
    necessary_feature_string_events = ["class_feature"]
    
    all_feature_strings_events = ["capital_letter_class_feature",  
                                  "word_class_feature", 
                                  "token_in_trigger_dict_class_feature", 
                                  "number_in_token_class_feature", 
                                  "word_stem_class_feature",
                                  "pos_class_feature",
                                  "token_is_after_dash_feature"]
    
    
    '''
    # new - testing optimal features
    (feature_strings_max, cm_max) = determine_optimal_features(stem_dict, word_dict, class_dict_events, trigger_dict, 
                                                      subsampled_tokens, 
                                                      test_sentences.tokens(), 
                                                      necessary_feature_string_events, 
                                                      list(all_feature_strings_events))
    
    
    '''
    
    '''
    # new optimal features error analysis
    feature_strings_max = ['class_feature', 'word_class_feature', 'pos_class_feature', 'number_in_token_class_feature'] 
    
    nb = NaiveBayes(stem_dict, word_dict, class_dict_events, trigger_dict)
    trained_nb = nb.train(subsampled_tokens, feature_strings_max)
    predictions = trained_nb.predict_all(test_sentences.tokens())
    
    y_test = map(lambda t: t.event_candidate, test_sentences.tokens())
    y_pred = predictions
    cm_max = confusion_matrix(class_dict_events, y_test, y_pred)
    
    print(cm_max)
    
    true_labels = []
    for token in test_sentences.tokens():
        true_labels.append(token.event_candidate)

    test_keys = class_dict_events.keys()
    test_keys.pop(0)
    for label in test_keys:
        print 'Analyzing label: ', label
        precision_recall_f1(true_labels, predictions, label)

    # Show confusion matrix in a separate window
    # import matplotlib.pyplot as plt
    plt.matshow(cm_max)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    '''
    
    '''
    
    # old ALL TRAINING
    nb = NaiveBayes(stem_dict, word_dict, class_dict, trigger_dict)
    
    trained_nb = nb.train(subsampled_tokens, ['class_feature', 'word_class_feature', 'capital_letter_class_feature'])
    
    # OLD ERROR ANALYSIS
    
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

    y_test = map(lambda t: t.event_candidate, all_test_tokens)
    y_pred = predictions

    # Compute confusion matrix
    cm = confusion_matrix(class_dict, y_test, y_pred)

    print(cm)

    # Show confusion matrix in a separate window
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
    
    '''
    
    necessary_feature_string_events_args = ["class_feature"]
    all_feature_strings_events_args = ["capital_letter_class_feature",  
                                       "word_class_feature", 
                                       "token_in_trigger_dict_class_feature", 
                                       "number_in_token_class_feature", 
                                       "word_stem_class_feature",
                                       "pos_class_feature",
                                       "token_is_after_dash_feature"]
    
    
    
    nb_args = NaiveBayesArgs(stem_dict, word_dict, class_dict_events_args, trigger_dict)
    
    trained_nb_args = nb_args.train(subsampled_tokens, ['class_feature', 'word_class_feature', 'capital_letter_class_feature'])
    
    y_test = []
    for token_parent in all_test_tokens:
        for token_child in token_parent.tokens_in_sentence:
            class_arg = "None"
            for (t_index, c) in token_parent.event_candidate_args:
                if t_index == token_parent.index:
                    class_arg = c
                    break
            y_test.append(class_arg)
    y_pred = trained_nb_args.predict_all(all_test_tokens)
    
    cm = confusion_matrix(class_dict_events_args, y_test, y_pred)
    
    print cm
    
    #for i in range(len(y_test)):
    #    print y_test[i], y_pred[i]
    
    #print y_test
    #print y_pred

def determine_optimal_features(stem_dict, word_dict, class_dict, trigger_dict, train_tokens, validation_tokens, neccessary_feature_strings, all_feature_strings):
    current_optimal_feature_strings = neccessary_feature_strings
    
    
    for i in range(len(all_feature_strings)):
        f_max = None
        f1_score_max = float("-inf")
        
        for f in all_feature_strings:
            nb = NaiveBayes(stem_dict, word_dict, class_dict, trigger_dict)
            
            temp_feature_strings = list(current_optimal_feature_strings)
            temp_feature_strings.append(f)
            
            #print 'Trying', temp_feature_strings
            
            trained_nb = nb.train(train_tokens, temp_feature_strings)
            predictions = trained_nb.predict_all(validation_tokens)
            
            y_test = map(lambda t: t.event_candidate, validation_tokens)
            y_pred = predictions
            
            #print y_test
            #print y_pred
            
            #print class_dict.keys()
            #print list(set(y_test))
            #print list(set(y_pred))
            
            cm = confusion_matrix(class_dict, y_test, y_pred)
            
            #print cm
            
            none_index = class_dict['None']
            
            f1_score = f1_micro(cm, none_index)
            if f1_score > f1_score_max:
                f1_score_max = f1_score
                f_max = f
                cm_max = cm
        
        if not f_max is None:
            current_optimal_feature_strings.append(f_max)
            if f_max in all_feature_strings:
                all_feature_strings.remove(f_max)
        
        print 'Optimal', current_optimal_feature_strings, 'with', f1_score_max
        print ""
        print ''
    
    return (current_optimal_feature_strings, cm_max)

if __name__ == "__main__":
    main()