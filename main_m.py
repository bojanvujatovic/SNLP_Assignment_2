from Classes.Sentences import Paragraphs
from Features.example_features import *
from functools import partial
from classifier.loglinear.Loglinear import LoglinearModel
from classifier.ErrorAnalysis import precision_recall_f1
from utils.Utils import *

def main_m():
    all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)

    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    trigger_dict = train_sentences.get_trigger_dict()
    stem_dict = get_stem_dict(train_sentences.tokens())

    print class_dict
    # print len(class_dict)
    # print trigger_dict


    all_tokens = []
    for sentence in train_sentences.sentences[0:10]:
        all_tokens.extend(sentence.tokens)
    print 'Before: ', len(all_tokens)
    all_tokens = subsample_none(all_tokens, 0.1)
    print 'After: ', len(all_tokens)
    
    phi = partial(set_of_features, stem_dict, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))

    feature_strings = ["word_template_feature", "word_class_template_feature",
                   "capital_letter_feature",
                   "token_in_trigger_dict_feature",
                   "number_in_token_feature",
                   "token_in_protein_feature",
                   "token_is_after_dash_feature",
                   "pos_class_feature",
                   "character_ngram_feature"]
    classifier = LoglinearModel(lambda w: w.event_candidate, phi, class_dict.keys(), 0.1, 10).train(
        all_tokens, feature_strings)

    # prediction = classifier.predict(train_sentences.sentences[0].tokens[0])
    # print 'Event label is: ', train_sentences.sentences[0].tokens[0].event_candidate
    # print 'Prediction is: ', prediction

    all_test_tokens = []
    for sentence in hand_out_sentences.sentences[0:5]:
        all_test_tokens.extend(sentence.tokens)

    predictions = classifier.predict_all(all_test_tokens, feature_strings)

    true_labels = []
    for token in all_test_tokens:
        true_labels.append(token.event_candidate)

    test_keys = class_dict.keys()
    test_keys.pop(0)
    for label in test_keys:
        print 'Analyzing label: ', label
        precision_recall_f1(true_labels, predictions, label)

if __name__ == "__main__":
    main_m()
