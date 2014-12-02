# -*- coding: utf-8 -*-

from scipy.sparse import csc_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class LoglinearModel(ClassifierModel):

    def __init__(self, gold, phi, classes, alpha, max_iterations=10):
        self.gold = gold
        self.phi = phi
        self.classes = classes
        self.alpha = alpha
        self.max_iterations = max_iterations

    def argmax(self, word, weights):
        max = (0.0, self.classes[0])
        for c in self.classes:
            c_prob = self.phi(word, c).T * weights
            if c_prob > max[0]:
                max = (c_prob, c)
        return max[1]

    def train(self, tokens):
        phi_length = self.phi(tokens[0], self.classes[0]).shape[0]
        weights = csc_matrix((phi_length, 1))

        for iterations in range(1, self.max_iterations):
            changed = False
            for token in tokens:
                prediction = self.argmax(token, weights)
                truth = self.gold(token)
                if truth != prediction:
                    weights += self.alpha * (self.phi(token, prediction) - self.phi(token, truth))
                    changed = True
            if not changed:
                break

        return TrainedLoglinearModel(weights, self)


class StructuredLoglinearModel(LoglinearModel):

    def argmax(self, word, weights):
        raise NotImplementedError


class TrainedLoglinearModel(TrainedClassifierModel):

    def __init__(self, weights, perceptron_model):
        self.__weights = weights
        self.__perceptron_model = perceptron_model

    def predict(self, token):
        return self.__perceptron_model.argmax(token, self.__weights)