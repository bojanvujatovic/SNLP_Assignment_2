# -*- coding: utf-8 -*-


class ClassifierModel(object):

    def train(self, tokens):
        raise NotImplementedError


class TrainedClassifierModel(object):

    def predict(self, token):
        raise NotImplementedError