# -*- coding: utf-8 -*-
import random
from numpy import argmax, mat, zeros
from classifier.ClassifierModel import ClassifierModel, TrainedClassifierModel


class StructuredLoglinearModel(ClassifierModel):
    def __init__(self, gold, phi, arguments, alpha, max_iterations=10):
        self.gold = gold
        self.phi = phi
        self.arguments = arguments
        self.alpha = alpha
        self.max_iterations = max_iterations

    # def get_argmax(self, word, weights):
    #     best_index = argmax([(self.phi(word, c).T * weights)[0, 0] for c in self.classes])
    #     return self.classes[best_index]

    def train(self, tokens):
        phi_length = self.phi((tokens[1], tokens[1]), self.arguments[1]).shape[0]
        weights = mat(zeros((phi_length, 1)))
        # TODO: average weights
        for epoch in range(1, self.max_iterations + 1):
            print 'epoch', epoch
            changed = False
            for token in tokens:
                truth = self.gold(token)
                for i in range(len(token.tokens_in_sentence)):
                    if truth[i] != 'None' or random.random() < 0.02:
                        best_index = argmax([(self.phi((token, token.tokens_in_sentence[i]), t).T * weights)[0, 0] for t in self.arguments])
                        prediction = self.arguments[best_index]
                        if prediction != truth[i]:
                            difference = self.phi((token, token.tokens_in_sentence[i]), truth[i]) - self.phi((token, token.tokens_in_sentence[i]), prediction)
                            weights = weights + self.alpha * difference
                            changed = True
            if not changed:
                break
        return TrainedStructuredLoglinearModel(weights, self)


class TrainedStructuredLoglinearModel(TrainedClassifierModel):
    def __init__(self, weights, perceptron_model):
        self.__weights = weights
        self.__perceptron_model = perceptron_model

    def predict(self, token):
        prediction = []
        for a in token.tokens_in_sentence:
            best_index = argmax([(self.__perceptron_model.phi((token, a), t).T * self.__weights)[0, 0] for t in self.__perceptron_model.arguments])
            prediction.append(self.__perceptron_model.arguments[best_index])

        return prediction

    def predict_all(self, list_of_tokens):
        return [self.predict(t) for t in list_of_tokens]



# class EarlyUpdateStructuredLoglinearModel(LoglinearModel):
#
#     def __init__(self, gold, phi, classes, arguments, alpha, max_iterations=10):
#         super(EarlyUpdateStructuredLoglinearModel, self).__init__(gold, phi, classes, alpha, max_iterations)
#         self.arguments = arguments
#
#     def get_argmax(self, word, weights):
#         truth = self.gold(word)
#         prediction = []
#         for i in range(len(word.sentence)):
#             best_index = argmax([(self.phi((word, word.sentence[i]), t).T * weights)[0, 0] for t in self.arguments])
#             if self.arguments[best_index] != truth[i]:
#                 break
#             else:
#                 prediction.append(self.arguments[best_index])
#         return prediction


# class NaiveStructuredLoglinearModel(LoglinearModel):
#
#     def __init__(self, gold, phi, classes, arguments, alpha, max_iterations=10):
#         super(NaiveStructuredLoglinearModel, self).__init__(gold, phi, classes, alpha, max_iterations)
#         self.arguments = arguments
#
#     def get_argmax(self, word, weights):
#         return [self.arguments[argmax([(self.phi(w, t).T * weights)[0, 0] for t in self.arguments])]
#                 for w in word.sentence]# if w != word]
#
#         # prediction = []
#         # for w in word.sentence:
#         #     if w == word:
#         #         continue
#         #
#         #     best_index = argmax([(self.phi(word, t).T * weights)[0, 0] for t in self.arguments])
#         #     prediction.append(self.arguments[best_index])
#         #
#         # return prediction


# class SearchStructuredLoglinearModel(LoglinearModel):
#
#     def __init__(self, gold, phi, classes, arguments, alpha, max_iterations=10):
#         super(SearchStructuredLoglinearModel, self).__init__(gold, phi, classes, alpha, max_iterations)
#         self.arguments = arguments
#
#     def get_argmax(self, word, weights):
#         pred = word.sentence.copy()
#         pred = map(lambda w: (w, 'None', (self.phi(w, 'None').T * weights)[0, 0], pred))
#
#         first = (0, 'None', float('-inf'))
#         second = (0, 'None', float('-inf'))
#         for i in range(len(pred)):
#             if pred[i][0] == word:
#                 continue
#             for arg in ['Cause', 'Theme']:
#                 diff = (self.phi(pred[i][0], arg).T * weights)[0, 0] - pred[i][2]
#                 if diff > first[2]:
#                     second = first
#                     first = (i, arg, diff)
#                 elif diff > second[2]:
#                     second = (i, arg, diff)
#
#         pred = map(lambda p: p[1], pred)
#         if first[2] > float('-inf'):
#             pred[first[0]] = first[1]
#         if second[2] > float('-inf'):
#             pred[second[0]] = second[1]


# class TrainedLoglinearModel(TrainedClassifierModel):
#     def __init__(self, weights, perceptron_model):
#         self.__weights = weights
#         self.__perceptron_model = perceptron_model
#
#     def predict(self, token):
#         return self.__perceptron_model.get_argmax(token, self.__weights)
#
#     def predict_all(self, list_of_tokens):
#         return [self.predict(t) for t in list_of_tokens]
