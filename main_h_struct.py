from Classes.Sentences import Paragraphs, Sentences
from Features.example_features import *
from functools import partial
from classifier.ErrorAnalysis import *
from classifier.loglinear.StructuredLoglinear import SearchStructuredLoglinearModel, StructuredLoglinearModel
from utils.Utils import *
import cloud.serialization.cloudpickle as cp
import numpy as np


def main():
    start = time.time()
    ### READ ###########################################################################################################
    print '\n------------'
    print 'Reading data'
    print '------------\n'

    all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()
    ###
    read_end = time.time()
    print 'Reading time:', read_end - start, 's'
    ####################################################################################################################

    ### PREPROCESS #####################################################################################################
    print '\n------------------'
    print 'Preprocessing data'
    print '------------------\n'

    used_fraction = 1
    train_fraction = 0.8
    none_fraction = 0.10

    print 'Fraction of data used:', used_fraction
    print 'Fraction of data for training:', train_fraction
    print 'Fraction of None-labelled samples used:', none_fraction

    (used_sentences, _) = all_train_sentences.split_randomly(used_fraction)
    (train_sentences, test_sentences) = used_sentences.split_randomly(train_fraction)

    all_train_tokens = train_sentences.tokens()
    subsampled_tokens = subsample_none(all_train_tokens, none_fraction)

    print 'Number of training tokens:', len(subsampled_tokens)

    class_dict = get_class_dict(subsampled_tokens)
    arg_dict = {'None': 0, 'Theme': 1, 'Cause': 2}
    stem_dict = get_stem_dict(subsampled_tokens)
    word_dict = get_word_dict(subsampled_tokens)
    ngram_order = 2
    char_ngram_dict = get_char_ngram_dict(subsampled_tokens, ngram_order)
    ngram_dict = get_ngram_dict(all_train_tokens, ngram_order)
    trigger_dict = get_trigger_dict(subsampled_tokens)
    arg_word_dict = get_arg_word_dict(subsampled_tokens)

    classes = dict(map(lambda c: (c, 0), class_dict.keys()))
    for token in subsampled_tokens:
        classes[token.event_candidate] += 1

    print classes

    feature_strings = [#'word_template_feature',
                       'word_class_template_feature',
                       'capital_letter_feature',
                       # 'token_in_trigger_dict_feature',
                       'number_in_token_feature',
                       'token_in_protein_feature',
                       # 'token_is_after_dash_feature',
                       'pos_class_feature']
                       # 'character_ngram_feature']
    phi = partial(set_of_features_structured, stem_dict, word_dict, arg_dict, class_dict, arg_word_dict, ngram_order, char_ngram_dict,
                  ngram_dict, feature_strings)

    print 'Used features:', feature_strings

    ###
    preprocess_end = time.time()
    print 'Preprocessing time:', preprocess_end - read_end, 's'
    ####################################################################################################################

    ### TRAIN ##########################################################################################################
    print '\n-------------'
    print 'Training data'
    print '-------------\n'

    alpha = 0.2
    max_iterations = 15
    arg_none_subsampling = 0.05

    def gold(trigger):
        args = [u'None'] * len(trigger.tokens_in_sentence)
        for (i, arg) in trigger.event_candidate_args:
            args[i] = arg
        return args

    print 'Alpha =', alpha
    print 'Max iterations =', max_iterations

    # classifier = SearchStructuredLoglinearModel(gold, phi, arg_dict.keys(), alpha, max_iterations)\
    #     .train(subsampled_tokens, average=True)

    classifier = StructuredLoglinearModel(gold, phi, arg_dict.keys(), alpha, arg_none_subsampling, max_iterations)\
        .train(subsampled_tokens, average=True)

    ###
    train_end = time.time()
    print 'Training time:', train_end - read_end, 's'
    ####################################################################################################################

    #### TEST ###########################################################################################################
    print '\n-------'
    print 'Testing'
    print '-------\n'

    all_test_tokens = test_sentences.tokens()
    subsampled_test_tokens = subsample_none(all_test_tokens, 0)

    print 'Number of test tokens:', len(subsampled_test_tokens)

    predictions = classifier.predict_all(subsampled_test_tokens)

    predict_end = time.time()
    print 'Predict time:', predict_end - train_end, 's'
    ####################################################################################################################

    ### ERROR ANALYSIS #################################################################################################
    print '\n-----------------'
    print 'Analysing results'
    print '-----------------\n'


    n_args = len(arg_dict)
    confusion = mat(zeros((n_args, n_args)))

    hits = 0
    misses = 0

    for i in range(0, len(predictions)):
        truth = gold(subsampled_test_tokens[i])
        if truth == predictions[i]:
            hits += 1
        else:
            misses += 1
        for j in range(0, len(predictions[i])):
            confusion[arg_dict[predictions[i][j]], arg_dict[truth[j]]] += 1

    np.set_printoptions(suppress=True)
    print confusion

    print 'precision micro:', precision_micro(confusion, 0)
    print 'recall micro:', recall_micro(confusion, 0)
    print 'f1 micro:', f1_micro(confusion, 0)

    print 'precision macro:', precision_macro(confusion, 0)
    print 'recall macro:', recall_macro(confusion, 0)
    print 'f1 macro:', f1_macro(confusion, 0)

    ###
    analysis_end = time.time()
    print '\nAnalysis time:', analysis_end - predict_end, 's'
    # ####################################################################################################################
    #
    cp.dump(classifier, open('classifier_' + time.strftime("%Y%m%d-%H%M%S") + '.p', 'wb'))


if __name__ == "__main__":
    main()
