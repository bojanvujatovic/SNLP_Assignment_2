# -*- coding: utf-8 -*-
from numpy import mat, zeros
import time
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class JointLoglinearModel(ClassifierModel):
    def __init__(self, gold, phi_trig, events, phi_arg, arguments, alpha, max_iterations=10):
        self.gold = gold
        self.phi_trig = phi_trig
        self.events = events
        self.phi_arg = phi_arg
        self.arguments = arguments
        self.alpha = alpha
        self.max_iterations = max_iterations

    def find_argmax(self, token, weights_arg, weights_trig):
        best_prediction = ('None', [])
        max_score = float('-inf')
        for e in self.events:
            arg_prediction = []
            if 'regulation' in e.lower():
                for c in token.tokens_in_sentence:
                    score_theme = self.phi_arg(c, 'Theme').T * weights_arg
                    score_none = self.phi_arg(c, 'None').T * weights_arg
                    if score_theme > score_none:
                        arg_prediction.append((score_theme, 'Theme'))
                    else:
                        arg_prediction.append((score_none, 'None'))
            else:
                for c in token.tokens_in_sentence:
                    best_arg = 'None'
                    best_score = float('-inf')
                    for a in self.arguments:
                        score = self.phi_arg(c, a).T * weights_arg
                        if score > best_score:
                            best_arg = a
                            best_score = score
                    arg_prediction.append((best_score, best_arg))
            score = self.phi_trig(token, e).T * weights_trig
            for (arg_score, _) in arg_prediction:
                score += arg_score
            if score > max_score:
                best_prediction = (e, map(lambda p: p[1], arg_prediction))
                max_score = score

        return best_prediction

    def train(self, tokens, average=False):
        phi_trig_length = self.phi_trig(tokens[0], self.events[0]).shape[0]
        weights_trig = mat(zeros((phi_trig_length, 1)))
        phi_arg_length = self.phi_trig(tokens[0], self.arguments[0]).shape[0]
        weights_arg = mat(zeros((phi_trig_length, 1)))
        for epoch in range(1, self.max_iterations + 1):
            print 'epoch', epoch
            changed = False
            for token in tokens:
                # Get argmax
                best_prediction = self.find_argmax(token, weights_arg, weights_trig)

                # Update weights
                truth = self.gold(token)
                if best_prediction != truth:
                    difference = self.phi_trig(token, truth[0]) - self.phi_trig(token, best_prediction[0])
                    weights_trig = weights_trig + self.alpha * difference
                    difference = 0.0
                    for i in range(len(token.tokens_in_sentence)):
                        difference += self.phi_trig(token.tokens_in_sentence[i], truth[1][i]) - \
                                      self.phi_trig(token.tokens_in_sentence[i], best_prediction[1][i])
                    weights_arg = weights_arg + self.alpha * difference
                    changed = True
            if not changed:
                break
        return TrainedJointLoglinearModel(weights_trig, weights_arg, self)


class TrainedJointLoglinearModel(TrainedClassifierModel):
    def __init__(self, weights_trig, weights_arg, loglinear_model):
        self.__weights_trig = weights_trig
        self.__weights_arg = weights_arg
        self.__loglinear_model = loglinear_model

    def predict(self, token):
        return self.__loglinear_model.find_argmax(token, self.__weights_trig, self.__weights_arg)

    def predict_all(self, list_of_tokens):
        prediction = list(list_of_tokens)
        start = time.time()
        for i in range(0, len(prediction)):
            prediction[i] = self.predict(list_of_tokens[i])
            if i % 500 == 0:
                print i, time.time() - start
                start = time.time()
        return prediction