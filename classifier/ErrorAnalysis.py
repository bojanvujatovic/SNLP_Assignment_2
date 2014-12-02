'''
Created on 2 Dec 2014

@author: miljan
'''

from Classes.Sentences import *
from classifier.loglinear.Loglinear import *
from sklearn.metrics.metrics import precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report

all_test_sentences = Paragraphs("Dataset/Test/").all_sentences()
all_test_tokens = [tokens for tokens in [sentence for sentence in all_test_sentences]]
true_candidates = [token.event_candidate for token in all_test_tokens]

if len(all_test_tokens != len(true_candidates)):
    print 'You messed up!'

class_dict = all_test_sentences.get_class_dict()
class_dict.pop('None', None)

# get classifier data, pickle it
predicted_candidates = []#predict_all()

print classification_report(true_candidates, predicted_candidates, 
                            labels=class_dict.values(),
                            target_names=class_dict.keys())

precision = precision_score(true_candidates, predicted_candidates)
recall = recall_score(true_candidates, predicted_candidates)
f1_score = f1_score(true_candidates, predicted_candidates)
confusion_matrix = confusion_matrix(true_candidates, predicted_candidates)

