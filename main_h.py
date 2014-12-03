from numpy import zeros, mat
from Classes.Sentences import Paragraphs
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

    # for t in subsampled_tokens:
    #     class_dict[t.event_candidate] += 1
    # print class_dict

    # phi = partial(word_belongs_to_class_feature, class_dict)
    phi = partial(word_stem, stem_dict)

    classifier = LoglinearModel(lambda t: t.event_candidate, phi, class_dict.keys(), 0.2, 10).train(subsampled_tokens)

    train_end = time.time()
    print 'training time:', train_end-start

    ####################################################################################################################
    ### TEST
    ####################################################################################################################


    predictions = classifier.predict_all(subsampled_tokens)

    predict_end = time.time()

    print 'predict time:', predict_end-train_end

    print predictions


if __name__ == "__main__":
    main()
