from Classes.Sentences import Paragraphs, Sentences
from Features.example_features import *
from functools import partial
from classifier.ErrorAnalysis import *
from classifier.loglinear.Loglinear import LoglinearModel
from utils.Utils import *
import cloud.serialization.cloudpickle as cp



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

    used_fraction = 0.5
    train_fraction = 0.8
    none_fraction = 0.1

    print 'Fraction of data used:', used_fraction
    print 'Fraction of data for training:', train_fraction
    print 'Fraction of None-labelled samples used:', none_fraction

    (used_sentences, _) = all_train_sentences.split_randomly(used_fraction)
    (train_sentences, test_sentences) = used_sentences.split_randomly(train_fraction)

    all_train_tokens = train_sentences.tokens()
    subsampled_tokens = subsample_none(all_train_tokens, none_fraction)

    print 'Number of training tokens:', len(subsampled_tokens)

    class_dict = get_class_dict(subsampled_tokens)
    stem_dict = get_stem_dict(subsampled_tokens)
    word_dict = get_word_dict(subsampled_tokens)
    ngram_order = 2
    ngram_dict = get_char_ngram_dict(subsampled_tokens, ngram_order)
    trigger_dict = get_trigger_dict(subsampled_tokens)

    # feature_strings = ["word_class_template_feature",
    #                    "capital_letter_feature",
    #                    "number_in_token_feature",
    #                    "character_ngram_feature"]
    feature_strings = ['word_template_feature',
                       'word_class_template_feature',
                       'capital_letter_feature',
                       'token_in_trigger_dict_feature',
                       'number_in_token_feature',
                       'token_in_protein_feature',
                       'token_is_after_dash_feature',
                       'pos_class_feature',
                       'character_ngram_feature']
    phi = partial(set_of_features, stem_dict, word_dict, class_dict, trigger_dict, ngram_order, ngram_dict, feature_strings)

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
    max_iterations = 10

    print 'Alpha =', alpha
    print 'Max iterations =', max_iterations

    classifier = LoglinearModel(lambda t: t.event_candidate, phi, class_dict.keys(), alpha, max_iterations)\
        .train(subsampled_tokens)

    ###
    train_end = time.time()
    print 'Training time:', train_end - read_end, 's'
    ####################################################################################################################

    #### TEST ###########################################################################################################
    print '\n-------'
    print 'Testing'
    print '-------\n'

    all_test_tokens = test_sentences.tokens()
    subsampled_test_tokens = all_test_tokens

    print 'Number of test tokens:', len(subsampled_test_tokens)

    predictions = classifier.predict_all(subsampled_test_tokens)

    ###
    predict_end = time.time()
    print 'Predict time:', predict_end - train_end, 's'
    ####################################################################################################################

    ### ERROR ANALYSIS #################################################################################################
    print '\n-----------------'
    print 'Analysing results'
    print '-----------------\n'

    true_labels = []
    for token in all_test_tokens:
        true_labels.append(token.event_candidate)

    test_keys = class_dict.keys()
    test_keys.pop(0)
    for label in test_keys:
        print 'Analyzing label: ', label
        precision_recall_f1(true_labels, predictions, label)

    # from sklearn.metrics import confusion_matrix
    import sklearn.metrics as sk

    y_test = map(lambda t: t.event_candidate, all_test_tokens)
    y_pred = predictions

    # Compute confusion matrix
    # cm = sk.confusion_matrix(y_test, y_pred)
    #
    # print(cm)

    cm2 = confusion_matrix(class_dict, y_test, y_pred)


    print cm2

    for i in range(len(class_dict)):
        print 'recall of', i, ':', label_recall(cm2, i)
        print 'precision of', i, ':', label_precision(cm2, i)
        print 'f1 of', i, ':', label_f1(cm2, i)
        
    print '\n'
    print 'precision micro:', precision_micro(cm2)
    print 'recall micro:', recall_micro(cm2)
    print 'f1 micro:', f1_micro(cm2)
    print '\n'
    print 'precision macro:', precision_macro(cm2)
    print 'recall macro:', recall_macro(cm2)
    print 'f1 macro:', f1_macro(cm2)
    

    # Show confusion matrix in a separate window
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    ###
    analysis_end = time.time()
    print 'Analysis time:', analysis_end - predict_end, 's'
    ####################################################################################################################

    cp.dump(classifier, open('classifier_' + time.strftime("%Y%m%d-%H%M%S") + '.p', 'wb'))
    # classifier = cp.loads(open('classifier.p', 'rb').read())


if __name__ == "__main__":
    main()
