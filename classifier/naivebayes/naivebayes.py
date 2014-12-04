# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel
from math import log

class NaiveBayes(ClassifierModel):

    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        self.class_dict = class_dict
        self.stem_dict = stem_dict
        self.word_dict = word_dict
        self.trigger_dict = trigger_dict

    def train(self, tokens, feature_strings):
        
        feature_counts = []
        for f in feature_strings:
            feature_class = globals()[f]
            feature_counts.append(feature_class(self.stem_dict, self.word_dict, self.class_dict, self.trigger_dict))
        
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
    
    def predict_all(self, list_of_tokens):
        return [self.predict(t) for t in list_of_tokens]

   
class feature(object):
    def __init__(self, class_dict, N_per_class):
        self.class_dict = class_dict
        self.counts = [[0 for j in range(N_per_class)] for i in range(len(class_dict))]
            
    def normalise_add_alpha_smothing(self, class_count, alpha):
        for class_string in class_count:
            class_id = self.class_dict[class_string]
                    
            for j in range(len(self.counts[class_id])):
                self.counts[class_id][j] = (self.counts[class_id][j] + alpha) / (float(class_count[class_string]) + len(class_count)*alpha)
    
class capital_letter_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(capital_letter_class_feature, self).__init__(class_dict, 1)
            
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
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(class_feature, self).__init__(class_dict, 1)
            
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
    
class word_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        word_dict["<<<<<UNK>>>>>"] = len(word_dict)
        
        super(word_class_feature, self).__init__(class_dict, len(word_dict))
        self.word_dict = word_dict
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        word_id  = self.word_dict[token.word]
        
        self.counts[class_id][word_id] += 1
        if self.counts[class_id][word_id] == 1:
            self.counts[class_id][self.word_dict["<<<<<UNK>>>>>"]] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        word_id  = self.word_dict.get(token.word, self.word_dict["<<<<<UNK>>>>>"])
        
        return self.counts[class_id][word_id]
    
class token_in_trigger_dict_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(token_in_trigger_dict_class_feature, self).__init__(class_dict, 1)
        self.trigger_dict = trigger_dict
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        
        if token.word in self.trigger_dict:
            self.counts[class_id][0] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        if token.word in self.trigger_dict:
            return self.counts[class_id][0]
        else:
            return 1 - self.counts[class_id][0]
        
class number_in_token_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(number_in_token_class_feature, self).__init__(class_dict, 1)
        self.trigger_dict = trigger_dict
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        
        if any([char.isdigit() for char in token.word]):
            self.counts[class_id][0] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        if any([char.isdigit() for char in token.word]):
            return self.counts[class_id][0]
        else:
            return 1 - self.counts[class_id][0]
        
class word_stem_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        stem_dict["<<<<<UNK>>>>>"] = len(stem_dict)
        
        super(word_stem_class_feature, self).__init__(class_dict, len(stem_dict))
        self.stem_dict = stem_dict
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        word_id  = self.stem_dict[token.stem]
        
        self.counts[class_id][word_id] += 1
        if self.counts[class_id][word_id] == 1:
            self.counts[class_id][self.stem_dict["<<<<<UNK>>>>>"]] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        word_id  = self.stem_dict.get(token.word, self.stem_dict["<<<<<UNK>>>>>"])
        
        return self.counts[class_id][word_id]
    
class pos_class_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(pos_class_feature, self).__init__(class_dict, 1)
        self.pos_tags = ['NN', 'NNP', 'NNS', 'NNPS',
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS']
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        
        if token.pos in self.pos_tags:
            self.counts[class_id][0] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        if token.pos in self.pos_tags:
            return self.counts[class_id][0]
        else:
            return 1 - self.counts[class_id][0]
        
class token_is_after_dash_feature(feature):
    def __init__(self, stem_dict, word_dict, class_dict, trigger_dict):
        super(token_is_after_dash_feature, self).__init__(class_dict, 1)
            
    def update(self, token, event_candidate):
        class_id = self.class_dict[event_candidate]
        
        if token.word[0] == "-":
            self.counts[class_id][0] += 1
    
    def prob(self, token, c):
        class_id = self.class_dict[c]
        if token.word[0] == "-":
            return self.counts[class_id][0]
        else:
            return 1 - self.counts[class_id][0]