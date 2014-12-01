'''Sentences.py
'''

from os   import walk
from json import JSONDecoder

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
                    
                    if json_paragraph.get("txt", None) != None and json_paragraph.get("sentences", None):
                        self.paragraphs.append(Paragraph(file_name, json_paragraph["txt"], json_paragraph["sentences"]))   
                    
        
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
        self.sentences = Sentences(self, sentences)

class Sentences(object):
    '''
    TODO: comment
    '''
    def __init__(self, paragraph, sentences):
        '''
        TODO: comment
        '''
        self.paragraph = paragraph
        self.sentences = []
        
        for sentence in sentences:
            self.sentences.append(Sentence(paragraph, sentence))
        
class Sentence(object):
    '''
    TODO: comment
    '''
    def __init__(self, paragraph, sentence):
        '''
        TODO: comment
        '''
        self.paragraph = paragraph
        self.tokens = []
        
        for token in sentence.get("tokens", None):
            index = token.get("index", None)
            word = token.get("word", None)
            
            if word == None:
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
                    
            event_candidate_args = [] 
            args = []
            
            print "index:", index
            
            for e in sentence.get("eventCandidates", None):
                print e.get("begin", None), "-", e.get("end", None)-1
                if index >= e.get("begin", None) and index < e.get("end", None):
                    print "poslije, uspilo uc!!!!"
                    args = e.get("arguments", None)
                    for arg in args:
                        for arg_index in range(arg.get("begin", None), arg.get("end", None)):
                            print "    ", arg.get("begin", None), arg.get("end", None)
                            arg_word = None
                            for token_t in sentence.get("tokens", None):
                                index_t = token_t.get("index", None)
                                word_t = token_t.get("word", None)
                                if index_t == arg_index:
                                    arg_word = word_t
                                    break
                            
                            event_candidate_args.append((arg_word, arg.get("gold", None)))
            print ""
            print event_candidate_args
            print ""
            print "======================================================="
            print""
                        
            self.tokens.append(Token(self, word, token.get("stem", None), index, 
                                     token.get("pos", None), token.get("begin", None), token.get("end", None), 
                                     mentions, deps, event_candidate, event_candidate_args))
            
            # zamijeniti deps indekse sa stvarnim rijecima za stv


class Token(object):
    '''
    TODO: comment
    '''
    def __init__(self, sentence, token, stem, index, pos, char_pos_begin, char_pos_end,
                 mentions, deps, event_candidate, event_candidate_args):
        '''
        TODO: comment
        '''
        self.sentence = sentence
        self.token = token
        self.stem = stem
        self.index = index
        self.pos = pos
        self.char_pos_begin = char_pos_begin
        self.char_pos_end = char_pos_end
        self.mentions = mentions
        self.deps = deps
        self.event_candidate = event_candidate
        self.event_candidate_args = event_candidate_args
        '''
        print sentence
        print token
        print stem
        print index
        print pos
        print char_pos_begin
        print char_pos_end
        print mentions
        print deps
        print event_candidate
        print event_candidate_args
        print ""
        print ""
        '''
