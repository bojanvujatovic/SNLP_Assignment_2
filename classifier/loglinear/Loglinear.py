# -*- coding: utf-8 -*-
from numpy import argmax, mat, zeros, copy

from scipy.sparse import csc_matrix, csr_matrix
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class LoglinearModel(ClassifierModel):
    def __init__(self, gold, phi, classes, alpha, max_iterations=10):
        self.gold = gold
        self.phi = phi
        self.classes = classes
        self.alpha = alpha
        self.max_iterations = max_iterations

    def get_argmax(self, word, weights, feature_strings):
        best_index = argmax([(self.phi(word, c, feature_strings).T * weights)[0, 0] for c in self.classes])
        return self.classes[best_index]

    def train(self, tokens, feature_strings):
        phi_length = self.phi(tokens[1], self.classes[1],feature_strings).shape[0]
        weights = mat(zeros((phi_length, 1)))

        for epoch in range(1, self.max_iterations + 1):
            print 'epoch', epoch
            changed = False
            for token in tokens:
                prediction = self.get_argmax(token, weights, feature_strings)
                truth = self.gold(token)
                if truth != prediction:
                    difference = self.phi(token, truth, feature_strings) - self.phi(token, prediction, feature_strings)
                    weights = weights + self.alpha * difference
                    changed = True
            if not changed:
                break
        print weights
        return TrainedLoglinearModel(weights, self)


class StructuredLoglinearModel(LoglinearModel):

    def __init__(self, gold, phi, classes, arguments, alpha, max_iterations=10):
        super(StructuredLoglinearModel, self).__init__(gold, phi, classes, alpha, max_iterations)
        self.arguments = arguments

    def get_argmax(self, word, weights):

        return [self.arguments[argmax([(self.phi(w, t).T * weights)[0, 0] for t in self.arguments])]
                for w in word.sentence if w != word]

        # prediction = []
        # for w in word.sentence:
        #     if w == word:
        #         continue
        #
        #     best_index = argmax([(self.phi(word, t).T * weights)[0, 0] for t in self.arguments])
        #     prediction.append(self.arguments[best_index])
        #
        # return prediction


class TrainedLoglinearModel(TrainedClassifierModel):
    def __init__(self, weights, perceptron_model):
        self.__weights = weights
        self.__perceptron_model = perceptron_model

    def predict(self, token, feature_strings):
        return self.__perceptron_model.get_argmax(token, self.__weights, feature_strings)
    
    def predict_all(self, list_of_tokens, feature_strings):
        return [self.predict(t, feature_strings) for t in list_of_tokens]
