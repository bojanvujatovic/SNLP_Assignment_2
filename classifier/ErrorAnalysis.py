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
# phi = partial(whole_set_of_features, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))
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

    print d_confusion_matrix

    precision = d_confusion_matrix['tp'] / (d_confusion_matrix['tp']+d_confusion_matrix['fp'])
    recall = d_confusion_matrix['tp'] / (d_confusion_matrix['tp']+d_confusion_matrix['fn'])
    f1_score = (2 * precision * recall) / (precision + recall)
    
    print '------------------'
    print 'For event label: ', event_label
    print 'Precision is: ', precision
    print 'Recall is: ', recall
    print 'F1 score is: ', f1_score
    print '------------------'
        
    
    
    