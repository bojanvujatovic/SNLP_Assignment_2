# '''
# Created on 2 Dec 2014
# 
# @author: miljan
# '''
# 
# from Classes.Sentences import *
# from classifier.loglinear import Loglinear
# from sklearn.metrics.metrics import precision_score, recall_score, f1_score, \
#     confusion_matrix, classification_report
# from Features.example_features import *
# from functools import partial
# 
# 
# all_test_sentences = Paragraphs("Dataset/Test/").all_sentences()
# all_test_tokens = [tokens for tokens in [sentence for sentence in all_test_sentences.sentences]]
# true_candidates = [token.event_candidate for token in all_test_tokens]
# 
# if (len(all_test_tokens) != len(true_candidates)):
#     print 'You messed up!'
# 
# all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
# (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)
# print train_sentences.sentences[0]
# 
# all_test_sentences = hand_out_sentences
# all_test_tokens = [tokens for tokens in [sentence for sentence in all_test_sentences.sentences]]
# true_candidates = [token.event_candidate for token in all_test_tokens]
# 
# word_dict = train_sentences.get_word_dict()
# class_dict = train_sentences.get_class_dict()
# trigger_dict = train_sentences.get_trigger_dict()
# 
# 
# phi = partial(whole_set_of_features, word_dict, class_dict, trigger_dict, 2, train_sentences.get_char_ngram_dict(2))
# 
# classifier = Loglinear.LoglinearModel(lambda w: w.event_candidate, phi, class_dict.keys(), 0.8, 10).train(
#     train_sentences.sentences[0].tokens)
# 
# 
# class_dict = all_test_sentences.get_class_dict()
# class_dict.pop('None', None)
# 
# # get classifier data, pickle it
# predicted_candidates = Loglinear.TrainedLoglinearModel.predict_all(classifier, all_test_tokens)
# 
# print classification_report(true_candidates, predicted_candidates, 
#                             labels=class_dict.values(),
#                             target_names=class_dict.keys())
# 
# precision = precision_score(true_candidates, predicted_candidates)
# recall = recall_score(true_candidates, predicted_candidates)
# f1_score = f1_score(true_candidates, predicted_candidates)
# confusion_matrix = confusion_matrix(true_candidates, predicted_candidates)


# Gives precision, recall and f1 score for the requested event
from numpy import mat, zeros
from numpy.core.umath import isnan


def precision_recall_f1(l_true_labels, l_predicted_labels, event_label):
    if (len(l_true_labels) != len(l_predicted_labels)):
        raise Exception('Input dimensions don\'t match')
    
    d_confusion_matrix = {'tp':0,'tn':0,'fp':0,'fn':0}
    # go through each token and update the confusion matrix
    for i in range(0, len(l_true_labels)):
        # true positives
        if l_true_labels[i] == event_label and l_predicted_labels[i] == event_label:
            d_confusion_matrix['tp'] += 1
        # true negatives
        elif l_true_labels[i] != event_label and l_predicted_labels[i] != event_label:
            d_confusion_matrix['tn'] += 1
        # false positivies
        elif l_true_labels[i] != event_label and l_predicted_labels[i] == event_label:
            d_confusion_matrix['fp'] += 1
        # false negatives
        elif l_true_labels[i] == event_label and l_predicted_labels[i] != event_label:
            d_confusion_matrix['fn'] += 1

    try:
        precision = d_confusion_matrix['tp'] / float(d_confusion_matrix['tp']+d_confusion_matrix['fp'])
    except:
        precision = 0
    try:
        recall = d_confusion_matrix['tp'] / float(d_confusion_matrix['tp']+d_confusion_matrix['fn'])
    except:
        recall = 0
    try:
        f1_score = (2 * precision * recall) / float(precision + recall)
    except:
        f1_score = 0

    print '------------------'
    print d_confusion_matrix
    print 'For event label: ', event_label
    print 'Precision is: ', precision
    print 'Recall is: ', recall
    print 'F1 score is: ', f1_score
    print '------------------'

    return d_confusion_matrix


# def precision_recall_f1_all(l_true_labels, l_predicted_labels, class_dict):
#     if (len(l_true_labels) != len(l_predicted_labels)):
#         raise Exception('Input dimensions don\'t match')
#
#     confusion_matrix = [[0 for x in range(10)] for x in range(10)]
#     relevant_labels = class_dict.keys()
#     relevant_labels.pop('None')
#     for event_label in relevant_labels:
#         for i in range(0, len(l_true_labels)):
#             # skip tp 'None-s'
#             if l_true_labels[i] == 'None' and l_predicted_labels[i] == 'None':
#                 continue
#             # true positives
#             if l_true_labels[i] == event_label and l_predicted_labels[i] == event_label:
#                 confusion_matrix[class_dict[event_label]][class_dict[event_label]] += 1
#             # true negatives
#             elif l_true_labels[i] != event_label and l_predicted_labels[i] != event_label:
#                 confusion_matrix[class_dict[l_true_labels[i]]][class_dict[event_label]] += 1
#             # false positives
#             elif l_true_labels[i] != event_label and l_predicted_labels[i] == event_label:
#                 d_confusion_matrix['fp'] += 1
#             # false negatives
#             elif l_true_labels[i] == event_label and l_predicted_labels[i] != event_label:
#                 d_confusion_matrix['fn'] += 1

def confusion_matrix(class_dict, y_test, y_pred):
    n_classes = len(class_dict)
    # n_classes = len(set(y_test))
    confusion = mat(zeros((n_classes, n_classes)))

    for i in range(0, len(y_pred)):
        confusion[class_dict[y_pred[i]], class_dict[y_test[i]]] += 1

    return confusion


def recall_micro(confusion, none_index, exclude_none=True):
    n = confusion.shape[0]
    num = sum([confusion[i, i] for i in range(n) if i != none_index or not exclude_none])
    den = sum([confusion[:, i].sum() for i in range(n) if i != none_index or not exclude_none])

    try:
        recall = float(num) / den
    except:
        recall = 0.0

    return recall if not isnan(recall) else 0.0


def precision_micro(confusion, none_index, exclude_none=True):
    n = confusion.shape[0]
    num = sum([confusion[i, i] for i in range(n) if i != none_index or not exclude_none])
    den = sum([confusion[i, :].sum() for i in range(n) if i != none_index or not exclude_none])
    try:
        precision = float(num) / den
    except:
        precision = 0.0

    return precision if not isnan(precision) else 0.0


def f1_micro(confusion, none_index, exclude_none=True):
    p = precision_micro(confusion, none_index, exclude_none)
    r = recall_micro(confusion, none_index, exclude_none)
    try:
        f1 = 2.0 * p * r / (p + r)
    except:
        f1 = 0.0

    return f1 if not isnan(f1) else 0.0


def label_precision(confusion, label):
    try:
        precision = float(confusion[label, label]) / confusion[label, :].sum()
    except:
        precision = 0.0

    return precision if not isnan(precision) else 0.0


def label_recall(confusion, label):
    try:
        recall = float(confusion[label, label]) / confusion[:, label].sum()
    except:
        recall = 0.0

    return recall if not isnan(recall) else 0.0


def label_f1(confusion, label):
    p = label_precision(confusion, label)
    r = label_recall(confusion, label)
    try:
        f1 = 2.0 * p * r / (p + r)
    except:
        f1 = 0.0

    return f1 if not isnan(f1) else 0.0


def precision_macro(confusion, none_index, exclude_none=True):
    n = confusion.shape[0]
    return sum([label_precision(confusion, i) for i in range(n) if i != none_index or not exclude_none]) / float(n)


def recall_macro(confusion, none_index, exclude_none=True):
    n = confusion.shape[0]
    return sum([label_recall(confusion, i) for i in range(n) if i != none_index or not exclude_none]) / float(n)


def f1_macro(confusion, none_index, exclude_none=True):
    p = precision_macro(confusion, none_index, exclude_none)
    r = recall_macro(confusion, none_index, exclude_none)
    try:
        f1 = 2.0 * p * r / (p + r)
    except:
        f1 = 0.0

    return f1 if not isnan(f1) else 0.0
