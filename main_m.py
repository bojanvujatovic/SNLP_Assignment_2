from Classes.Sentences import Paragraphs, Sentences
from Features.example_features import *
from functools import partial
from classifier.loglinear.Loglinear import LoglinearModel
from classifier.ErrorAnalysis import precision_recall_f1
from utils.Utils import *

def main_m():
    return
    # all_train_sentences = Paragraphs("Dataset/Train/").all_sentences()
    #
    # # word_dict = all_train_sentences[0:10].get_word_dict()
    # # class_dict = all_train_sentences[0:10].get_class_dict()
    # # trigger_dict = all_train_sentences[0:10].get_trigger_dict()
    # # stem_dict = get_stem_dict(all_train_sentences[0:10].tokens())
    #
    # all_train_sentences = all_train_sentences.sentences[0:10]
    # all_tokens = all_train_sentences.tokens()
    # subsampled_tokens = subsample_none(all_tokens, 0.1)
    # class_dict = get_class_dict(subsampled_tokens)
    # stem_dict = get_stem_dict(subsampled_tokens)
    # word_dict = get_word_dict(subsampled_tokens)
    #
    # s = Sentences("", [{"tokens": []}])
    # s.sentences[0].tokens += subsampled_tokens
    # trigger_dict = s.get_trigger_dict()
    #
    # print class_dict
    #
    # all_tokens = []
    # for sentence in all_train_sentences.sentences[0:10]:
    #     all_tokens.extend(sentence.tokens)
    # print 'Before: ', len(all_tokens)
    # all_tokens = subsample_none(all_tokens, 0.05)
    # print 'After: ', len(all_tokens)
    #
    # feature_strings = ["word_class_template_feature",
    #                    "capital_letter_feature",
    #                    "number_in_token_feature"]
    # phi = partial(set_of_features, stem_dict, word_dict, class_dict, trigger_dict, 2, all_train_sentences.get_ngram_dict(2),
    #               feature_strings)
    #
    # classifier = LoglinearModel(lambda t: t.event_candidate, phi, class_dict.keys(), 0.2, 10).train(subsampled_tokens)
    #
    #
    # # ---- Testing ----
    #
    # all_test_sentences = Paragraphs("Dataset/Test/").all_sentences()
    # all_test_tokens = []
    # for sentence in all_test_sentences.sentences[0:5]:
    #     all_test_tokens.extend(sentence.tokens)
    # print len(all_test_tokens)
    # all_test_tokens = subsample_none(all_test_tokens, 0.05)
    # print len(all_test_tokens)
    #
    # predictions = classifier.predict_all(all_test_tokens)
    #
    # true_labels = []
    # for token in all_test_tokens:
    #     true_labels.append(token.event_candidate)
    #
    # for label in class_dict.keys():
    #     if label == 'None':
    #         continue
    #     print 'Analyzing label: ', label
    #     precision_recall_f1(true_labels, predictions, label)

if __name__ == "__main__":
    main_m()
