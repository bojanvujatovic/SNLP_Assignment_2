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
            # print self.phi(word, c).shape
            # print weights.shape
            c_prob = self.phi(word, c).T * weights
            if c_prob > max[0]:
                max = (c_prob, c)
        return max[1]

    def train(self, tokens):
        phi_length = self.phi(tokens[1], self.classes[1]).shape[0]
        weights = csc_matrix((phi_length, 1))

        for iterations in range(1, self.max_iterations):
            changed = False
            for token in tokens:
                prediction = self.argmax(token, weights)
                truth = self.gold(token)
                if truth != prediction:
                    # print prediction
                    # print truth
                    weights = weights + self.alpha * (self.phi(token, prediction) - self.phi(token, truth))
                    changed = True
            if not changed:
                break

        return TrainedLoglinearModel(weights, self)


class StructuredLoglinearModel(LoglinearModel):
    # if label(x) == None: no arguments
    # if label(x) != None: at least 1 theme
    # if label(x) == Regulation: there can exist cause argument

    def argmax(self, word, weights):
        arg_types = ['None', 'Theme', 'Cause']
        prediction = []
        for w in word.sentence:
            if w == word:
                continue

            prob = (0.0, arg_types[0])
            for t in arg_types:
                t_prob = self.phi(w, t).T * weights
                if t_prob > prob[0]:
                    prob = (t_prob, t)

            prediction.append(prob[1])
        return prediction


class TrainedLoglinearModel(TrainedClassifierModel):
    def __init__(self, weights, perceptron_model):
        self.__weights = weights
        self.__perceptron_model = perceptron_model

    def predict(self, token):
        return self.__perceptron_model.argmax(token, self.__weights)
    
    def predict_all(self, list_of_tokens):
        return [self.predict(t) for t in list_of_tokens]
