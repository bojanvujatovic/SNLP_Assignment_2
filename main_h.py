from numpy import zeros, mat
from Classes.Sentences import *
from Features.example_features import *
from functools import partial
from classifier.loglinear.Loglinear import LoglinearModel
import time
from utils.Utils import *


def main():
    all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)


    ####################################################################################################################
    ### TRAIN
    ####################################################################################################################

    start = time.time()

    all_tokens = train_sentences.tokens()
    subsampled_tokens = subsample_none(all_tokens, 0.0)
    subsampled_tokens = subsample_label(subsampled_tokens, 'Gene_expression', 0.15)
    class_dict = get_class_dict(subsampled_tokens)
    stem_dict = get_stem_dict(subsampled_tokens)
    word_dict = get_word_dict(subsampled_tokens)
    
    s = Sentences("", [{"tokens": []}])
    s.sentences[0].tokens += subsampled_tokens
    trigger_dict = s.get_trigger_dict()

    # for t in subsampled_tokens:
    #     class_dict[t.event_candidate] += 1
    # print class_dict

    phi = partial(set_of_features, stem_dict, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))
    
    feature_strings = ["word_class_template_feature", "capital_letter_feature", "number_in_token_feature"]
    classifier = LoglinearModel(lambda t: t.event_candidate, phi, class_dict.keys(), 0.2, 10).train(subsampled_tokens, feature_strings)

    train_end = time.time()
    print 'training time:', train_end-start

    ####################################################################################################################
    ### TEST
    ####################################################################################################################


    predictions = classifier.predict_all(subsampled_tokens, feature_strings)

    predict_end = time.time()

    print 'predict time:', predict_end-train_end

    print predictions


if __name__ == "__main__":
    main()
