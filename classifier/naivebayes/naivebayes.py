# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class NaiveBayes(ClassifierModel):

    def __init__(self, phi, class_dict):
        self.phi = phi
        self.class_dict = class_dict

    def train(self, tokens):
        phi_length = self.phi(tokens[0], self.class_dict.keys()[0]).shape[0]
        counts = [csr_matrix((phi_length, 1)) for i in range(len(self.class_dict))]
        class_counts = [0.0] * len(self.class_dict)
        
        N = len(tokens)
        i = 0
        
        for token in tokens:
            i += 1
            
            if i % 1000 == 0:
                print i, "/", N
            
            class_idx = self.class_dict[token.event_candidate]
            class_counts[class_idx] += 1
            token_phi = self.phi(token, token.event_candidate)
            
            counts[class_idx] = counts[class_idx] + token_phi
        
        for i in range(len(self.class_dict)):
            counts[i] = counts[i] * 1.0/class_counts[i]

        print counts[0]
        return TrainedNaiveBayes(counts)

class TrainedNaiveBayes(TrainedClassifierModel):

    def __init__(self, probabilities):
        self.__probabilities = probabilities

    def predict(self, token):
        return 0