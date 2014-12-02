from Classes.Sentences import Paragraphs
from Features.example_features import word_class_template_feature


def main():

    all_train_sentences = Paragraphs("Dataset/Train_small/").all_sentences()
    (train_sentences, hand_out_sentences) = all_train_sentences.split_randomly(0.8)

    word_dict = train_sentences.get_word_dict()
    class_dict = train_sentences.get_class_dict()
    trigger_dict = train_sentences.get_trigger_dict()

    print class_dict
    print trigger_dict

    word_class_template_feature(train_sentences.sentences[1].tokens[1], "None", None, word_dict, class_dict)




if __name__ == "__main__":
    main()
