'''Sentences.py
'''

from os import walk
from json import JSONDecoder

import random


class Paragraphs(object):
    '''
    TODO: comment
    '''

    def __init__(self, directory):
        '''
        TODO: comment
        '''
        self.paragraphs = []

        decoder = JSONDecoder()

        for (root, _, files) in walk(directory):
            for file_name in files:

                if not file_name.endswith(".json"):
                    continue

                file_path = root + "/" + file_name

                with open(file_path, 'r') as f:
                    file_content = f.read()
                    json_paragraph = decoder.decode(file_content)

                    if not json_paragraph.get("txt", None) is None and json_paragraph.get("sentences", None):
                        self.paragraphs.append(Paragraph(file_name, json_paragraph["txt"], json_paragraph["sentences"]))

    def all_sentences(self):
        ret = Sentences("", [])
        for paragraph in self.paragraphs:
            ret.sentences += paragraph.sentences.sentences

        return ret


class Paragraph(object):
    '''
    TODO: comment
    '''

    def __init__(self, name, txt, sentences):
        '''
        TODO: comment
        '''
        self.name = name
        self.txt = txt
        self.sentences = Sentences(txt, sentences)


class Sentences(object):
    '''
    TODO: comment
    '''

    def __init__(self, paragraph_txt, sentences):
        '''
        TODO: comment
        '''
        self.sentences = []

        for sentence in sentences:
            self.sentences.append(Sentence(paragraph_txt, sentence))

    def tokens(self):
        return [t for s in self.sentences for t in s.tokens]

    def split_randomly(self, split_percentage):
        A = Sentences("", [])
        B = Sentences("", [])

        random.seed(1)
        for sentence in self.sentences:
            if random.random() < split_percentage:
                A.sentences.append(sentence)
            else:
                B.sentences.append(sentence)

        return (A, B)

    def get_word_dict(self):
        ret = {}
        counter = 0

        for sentence in self.sentences:
            for token in sentence.tokens:
                if ret.get(token.word, None) is None:
                    ret[token.word] = counter
                    counter += 1
        ret['<<UNK>>'] = counter
        return ret

    def get_class_dict(self):
        ret = {}
        counter = 0

        for sentence in self.sentences:
            for token in sentence.tokens:
                if ret.get(token.event_candidate, None) is None and token.event_candidate is not None:
                    ret[token.event_candidate] = counter
                    counter += 1

        return ret
    
    def get_class_args_dict(self):
        ret = {}
        counter = 0

        for sentence in self.sentences:
            for token in sentence.tokens:
                for (_, class_arg) in token.event_candidate_args:
                    if ret.get(class_arg, None) is None and class_arg is not None:
                        ret[class_arg] = counter
                        counter += 1

        return ret

    def get_trigger_dict(self):
        ret = {}
        counter = 0

        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.event_candidate != "None" and not token.event_candidate is None and ret.get(
                        token.event_candidate, None) is None:
                    ret[token.word] = counter
                    counter += 1

                    if '-' in token.word:
                        for word_t in token.word.split('-'):
                            if ret.get(word_t, None) is None and len(word_t) > 0:
                                ret[word_t] = counter
                                counter += 1
        return ret

    def get_ngram_dict(self, n):
        ret = {}
        counter = 0

        for sentence in self.sentences:
            for token in sentence.tokens:
                ngrams = [token.word[i:i + n] for i in range(len(token.word) - n + 1)]
                for key in ngrams:
                    if key not in ret:
                        ret[key] = counter
                        counter += 1
        ret['<<UNK>>'] = counter
        return ret


class Sentence(object):
    '''
    TODO: comment
    '''

    def __init__(self, paragraph_txt, sentence):
        '''
        TODO: comment
        '''
        self.paragraph_txt = paragraph_txt
        self.tokens = []

        sentence_index = 0

        for token in sentence.get("tokens", None):
            index = token.get("index", None)
            word = token.get("word", None)

            if word is None:
                continue

            mentions = []
            for mention in sentence.get("mentions", None):
                if index >= mention.get("begin", None) and index < mention.get("end", None):
                    mentions.append(mention.get("label", None))

            deps = []
            for dep in sentence.get("deps", None):
                if index == dep.get("mod", None):
                    head = None
                    for token_t in sentence.get("tokens", None):
                        index_t = token_t.get("index", None)
                        word_t = token_t.get("word", None)
                        if index_t == dep.get("head", None):
                            head = word_t

                    deps.append((head, "head", dep.get("label", None)))

                if index == dep.get("head", None):
                    mod = None
                    for token_t in sentence.get("tokens", None):
                        index_t = token_t.get("index", None)
                        word_t = token_t.get("word", None)
                        if index_t == dep.get("mod", None):
                            mod = word_t

                    deps.append((mod, "mod", dep.get("label", None)))

            event_candidate = None
            for e in sentence.get("eventCandidates", None):
                if index >= e.get("begin", None) and index < e.get("end", None):
                    event_candidate = e.get("gold", None)
                    break
            if event_candidate == None:
                event_candidate = u'None'

            event_candidate_args = []
            args = []

            for e in sentence.get("eventCandidates", None):
                if index >= e.get("begin", None) and index < e.get("end", None):
                    args = e.get("arguments", None)
                    for arg in args:
                        for arg_index in range(arg.get("begin", None), arg.get("end", None)):
                            event_candidate_args.append((arg_index, arg.get("gold", None)))

            self.tokens.append(Token(self, word, token.get("stem", None), index,
                                     token.get("pos", None), token.get("begin", None), token.get("end", None),
                                     mentions, deps, event_candidate, event_candidate_args, sentence_index))

            sentence_index += 1


class Token(object):
    '''
    TODO: comment
    '''

    def __init__(self, sentence, word, stem, index, pos, char_pos_begin, char_pos_end,
                 mentions, deps, event_candidate, event_candidate_args, sentence_index):
        '''
        TODO: comment
        '''
        self.tokens_in_sentence = sentence.tokens
        self.word = word
        self.stem = stem
        self.index = index
        self.pos = pos
        self.char_pos_begin = char_pos_begin
        self.char_pos_end = char_pos_end
        self.mentions = mentions
        self.deps = deps

        # What we want to learn
        self.event_candidate = event_candidate
        self.event_candidate_args = event_candidate_args
            
        # Derived features
        self.paragraph_text = sentence.paragraph_txt
        self.sentence_index = sentence_index

        '''   
        print ""
        print ""
        
        
        print self.tokens_in_sentence 
        print self.word 
        print self.stem 
        print self.index
        print self.pos
        print self.char_pos_begin 
        print self.char_pos_end 
        print self.mentions
        print self.deps 
        
        # What we want to learn
        print self.event_candidate 
        print self.event_candidate_args 
        
        # Derived features
        print self.paragraph_text 
        print self.sentence_index
        ''' 
