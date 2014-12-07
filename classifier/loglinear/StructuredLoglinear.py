# -*- coding: utf-8 -*-
import random
from numpy import argmax, mat, zeros
import time
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class SearchStructuredLoglinearModel(ClassifierModel):
    def __init__(self, gold, phi, arguments, alpha, max_iterations=10):
        self.gold = gold
        self.phi = phi
        self.arguments = arguments
        self.alpha = alpha
        self.max_iterations = max_iterations

    def train(self, tokens, average=False):
        phi_length = self.phi((tokens[1], tokens[1]), self.arguments[1]).shape[0]
        weights = mat(zeros((phi_length, 1)))
        avg_w = weights.copy()
        counter = 0
        for epoch in range(1, self.max_iterations + 1):
            print 'epoch', epoch
            changed = False
            for token in tokens:
                # Get argmax
                truth = self.gold(token)
                pred = list(token.tokens_in_sentence)
                pred = map(lambda w: (w, 'None', (self.phi((token, w), 'None').T * weights)[0, 0]), pred)

                first = (0, 'None', float('-inf'))
                second = (0, 'None', float('-inf'))
                for i in range(len(pred)):
                    if pred[i][0] == token:
                        continue
                    for arg in ['Cause', 'Theme']:
                        diff = (self.phi((token, pred[i][0]), arg).T * weights)[0, 0] - pred[i][2]
                        if diff > first[2]:
                            second = first
                            first = (i, arg, diff)
                        elif diff > second[2]:
                            second = (i, arg, diff)

                pred = map(lambda p: p[1], pred)
                if first[2] > float('-inf'):
                    pred[first[0]] = first[1]
                if second[2] > float('-inf'):
                    pred[second[0]] = second[1]

                # Update weights
                for i in range(len(token.tokens_in_sentence)):
                    if pred[i] != truth[i]:
                        arg_cand = token.tokens_in_sentence[i]
                        difference = self.phi((token, arg_cand), truth[i]) - self.phi((token, arg_cand), pred[i])
                        weights = weights + self.alpha * difference
                        avg_w = avg_w + weights
                        counter += 1
                        changed = True
            if not changed:
                break
        if average:
            return TrainedStructuredLoglinearModel(avg_w, self)
        else:
            return TrainedStructuredLoglinearModel(weights, self)


class StructuredLoglinearModel(ClassifierModel):
    def __init__(self, gold, phi, arguments, alpha, arg_none_subsampling, max_iterations=10):
        self.arg_none_subsampling = arg_none_subsampling
        self.gold = gold
        self.phi = phi
        self.arguments = arguments
        self.alpha = alpha
        self.max_iterations = max_iterations

    def train(self, tokens, average=False):
        phi_length = self.phi((tokens[1], tokens[1]), self.arguments[1]).shape[0]
        weights = mat(zeros((phi_length, 1)))
        avg_w = weights.copy()
        counter = 0
        for epoch in range(1, self.max_iterations + 1):
            print 'epoch', epoch
            changed = False
            for token in tokens:
                truth = self.gold(token)
                for i in range(len(token.tokens_in_sentence)):
                    if truth[i] != 'None' or random.random() < self.arg_none_subsampling:
                        arg_cand = token.tokens_in_sentence[i]
                        best_index = argmax([(self.phi((token, arg_cand), t).T * weights)[0,0] for t in self.arguments])
                        prediction = self.arguments[best_index]
                        if prediction != truth[i]:
                            difference = self.phi((token, arg_cand), truth[i]) - self.phi((token, arg_cand), prediction)
                            weights = weights + self.alpha * difference
                            avg_w = avg_w + weights
                            counter += 1
                            changed = True
            if not changed:
                break
        if average:
            return TrainedStructuredLoglinearModel(avg_w, self)
        else:
            return TrainedStructuredLoglinearModel(weights, self)


class TrainedStructuredLoglinearModel(TrainedClassifierModel):
    def __init__(self, weights, loglinear_model):
        self.__weights = weights
        self.__loglinear_model = loglinear_model

    def predict(self, token):
        prediction = []
        for a in token.tokens_in_sentence:
            best_index = argmax([(self.__loglinear_model.phi((token, a), t).T * self.__weights)[0, 0]
                                 for t in self.__loglinear_model.arguments])
            prediction.append(self.__loglinear_model.arguments[best_index])

        return prediction

    def predict_all(self, list_of_tokens):
        prediction = list(list_of_tokens)
        start = time.time()
        for i in range(0, len(prediction)):
            prediction[i] = self.predict(list_of_tokens[i])
            if i % 500 == 0:
                print i, time.time() - start
                start = time.time()
        return prediction