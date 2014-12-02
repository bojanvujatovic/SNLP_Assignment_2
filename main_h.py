from Classes.Sentences import Paragraphs
from Features.example_features import *
from functools import partial
from classifier.loglinear.Loglinear import LoglinearModel


def main():
    all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)

    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    trigger_dict = train_sentences.get_trigger_dict()

    # print class_dict
    # print len(class_dict)
    # print trigger_dict

    phi = partial(whole_set_of_features, word_dict, class_dict, trigger_dict, 2, train_sentences.get_ngram_dict(2))

    classifier = LoglinearModel(lambda w: w.event_candidate, phi, class_dict.keys(), 0.8, 10).train(
        train_sentences.sentences[0].tokens)

    prediction = classifier.predict(train_sentences.sentences[0].tokens[0])

    print prediction


if __name__ == "__main__":
    main()
