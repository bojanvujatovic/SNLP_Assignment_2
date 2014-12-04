# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel
from math import log

class NaiveBayes(ClassifierModel):

    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict, n, ngram_combinations):
        self.class_dict = class_dict
        self.stem_dict = stem_dict
        self.word_dict = word_dict
        self.trigger_dict = trigger_dict
        self.n = n
        self.ngram_combinations = ngram_combinations

    def train(self, tokens, feature_strings):
        
        feature_counts = []
        for f in feature_strings:
            feature_class = globals()[f]
            feature_counts.append(feature_class(self.stem_dict, self.word_dict, self.class_dict, self.trigger_dict, self.n, self.ngram_combinations))
        
        class_count = {}
        for c in self.class_dict:
            class_count[c] = 0
        
        for f in feature_counts:
            for token in tokens:
                f.update(token, token.event_candidate)
                if f == feature_counts[0]: # just once
                    class_count[token.event_candidate] += 1
            f.normalise_add_alpha_smothing(class_count, 0.001)

        return TrainedNaiveBayes(feature_counts, self.class_dict)
        

class TrainedNaiveBayes(TrainedClassifierModel):

    def __init__(self, feature_probs, class_dict):
        self.feature_probs = feature_probs
        self.class_dict = class_dict

    def predict(self, token):
        log_prod_max = float("-inf")
        
        for c in self.class_dict:
            log_prod = 0.0
            for f in self.feature_probs:
                log_prod += log(f.prob(token, c))
            
            if log_prod > log_prod_max:
                log_prod_max = log_prod
                c_max = c
        
        return c_max
    
class feature(object):
    def __init__(self, class_dict):
        self.class_dict = class_dict
        self.counts = [[0] for i in range(len(class_dict))]
            
    def normalise_add_alpha_smothing(self, class_count, alpha):
        for class_string in class_count:
            class_id = self.class_dict[class_string]
                    
            for j in range(len(self.counts[class_id])):
                self.counts[class_id][j] = (self.counts[class_id][j] + alpha) / (float(class_count[class_string]) + len(class_count)*alpha)
    
class capital_letter_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict, n, ngram_combinations):
        super(capital_letter_feature, self).__init__(class_dict)
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        
        if token.word[0].isupper():
            self.counts[class_id][0] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        if token.word[0].isupper():
            return self.counts[class_id][0]
        else:
            return 1 - self.counts[class_id][0]
            
class class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict, n, ngram_combinations):
        super(class_feature, self).__init__(class_dict)
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        self.counts[class_id][0] += 1 
        
    def normalise_add_alpha_smothing(self, class_count, alpha):        
        for class_string in class_count:
            class_id = self.class_dict[class_string]
                    
            for j in range(len(self.counts[class_id])):
                self.counts[class_id][j] = (self.counts[class_id][j] + alpha) / (sum(class_count.values()) + len(class_count)*alpha)
        
    def prob(self, token, c):
        class_id = self.class_dict[c]
        return self.counts[class_id][0]