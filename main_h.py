import numpy as np
from scipy import *
from Classes.Sentences import Paragraphs, Sentences
from Features.example_features import *
from functools import partial
from classifier.ErrorAnalysis import precision_recall_f1
from classifier.loglinear.Loglinear import LoglinearModel
import time
from utils.Utils import *
import pickle


def phi(word, c):
    print 'phi(', word, c, ')'
    x = np.mat(np.zeros((2, 1)))
    if word[0] == 'increase' and c == 'regulation':
        print 'case 1'
        x[0, 0] = 1
    elif word[0] == 'expression' and c == 'Gene_expression':
        print 'case 2'
        x[1, 0] = 1
    return x


def main():
    ####################################################################################################################
    ### TRAIN
    ####################################################################################################################

    start = time.time()

    fraction_of_data = 0.1

    all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()

    ###
    read_end = time.time()
    print 'reading time:', read_end - start
    ###

    (train_sentences, non_train_sentences) = all_train_sentences.split_randomly(fraction_of_data)
    (test_sentences, shit_sentences) = all_train_sentences.split_randomly(fraction_of_data)

    print len(train_sentences.sentences)


    all_tokens = train_sentences.tokens()
    subsampled_tokens = subsample_none(all_tokens, 0.07)
    print (len(all_tokens), len(subsampled_tokens))


    class_dict = get_class_dict(subsampled_tokens)
    stem_dict = get_stem_dict(subsampled_tokens)
    word_dict = get_word_dict(subsampled_tokens)
    ngram_dict = get_ngram_dict(subsampled_tokens, 2)

    s = Sentences("", [{"tokens": []}])
    s.sentences[0].tokens += subsampled_tokens
    trigger_dict = s.get_trigger_dict()

    # for t in subsampled_tokens:
    # class_dict[t.event_candidate] += 1
    # print class_dict

    feature_strings = ["word_class_template_feature",
                       "capital_letter_feature",
                       "number_in_token_feature",
                       "character_ngram_feature"]
    phi = partial(set_of_features, stem_dict, word_dict, class_dict, trigger_dict, 2, ngram_dict, feature_strings)

    ###
    preprocess_end = time.time()
    print 'preprocessing time:', preprocess_end - read_end
    ###

    classifier = LoglinearModel(lambda t: t.event_candidate, phi, class_dict.keys(), 0.2, 10).train(subsampled_tokens)

    ###
    train_end = time.time()
    print 'training time:', train_end - read_end
    ###

    ####################################################################################################################
    ### TEST
    ####################################################################################################################


    predictions = classifier.predict_all(subsampled_tokens)

    ###
    predict_end = time.time()
    print 'predict time:', predict_end - train_end
    ###

    print predictions

    all_test_tokens = []
    for sentence in test_sentences.sentences:
        all_test_tokens.extend(sentence.tokens)

    all_test_tokens = subsample_none(all_test_tokens, 0)

    predictions = classifier.predict_all(all_test_tokens)
    print predictions
    print len(predictions)

    true_labels = []
    for token in all_test_tokens:
        true_labels.append(token.event_candidate)

    test_keys = class_dict.keys()
    test_keys.pop(0)
    for label in test_keys:
        print 'Analyzing label: ', label
        precision_recall_f1(true_labels, predictions, label)

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    y_test = map(lambda t: t.event_candidate, all_test_tokens)
    y_pred = predictions

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # pickle.dump(classifier, open('classifier.p', 'rb'))


if __name__ == "__main__":
    main()
