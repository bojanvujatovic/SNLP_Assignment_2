# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class NaiveBayes(ClassifierModel):

    def __init__(self, gold, phi, classes):
        self.gold = gold
        self.phi = phi
        self.classes = classes

    def argmax(self, word, weights):
        max = (0.0, self.classes[0])
        for c in self.classes:
            c_prob = self.phi(word, c).T * weights
            if c_prob > max[0]:
                max = (c_prob, c)
        return max[1]

    def train(self, tokens):
        phi_length = self.phi(tokens[0], self.classes[0]).shape[0]
        counts = [csr_matrix((phi_length, 1)) for i in range(len(self.classes))]
        class_counts = [0.0] * len(self.classes)
        
        for token in tokens:
            class_idx = self.classes[token.event_candidate]
            class_counts[class_idx] += 1
            token_phi = self.phi(token, token.event_candidate)
            
            counts[class_idx] = counts[class_idx] + token_phi
        
        for i in range(len(self.classes)):
            counts[i] = counts[i] * 1.0/class_counts[i]

        print counts
        return TrainedLoglinearModel(counts)

class TrainedNaiveBayes(TrainedClassifierModel):

    def __init__(self, probabilities):
        self.__probabilities = probabilities

    def predict(self, token):
        return 0