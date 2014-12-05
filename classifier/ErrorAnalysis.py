'''
Created on 2 Dec 2014

@author: miljan
'''



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
