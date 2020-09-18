# Generators yielding words for n-grams, sentences for word2vec

def group1(matchobj):
    return matchobj.group(1)

def group2(matchobj):
    return matchobj.group(2)

def clean_token(t, remchars = "[`¬@^\|\"]"):
    '''
    Cleansing of word token strings using regex matching.
    Includes several elements which are wiki-specific, based on removing wiki markup.
    
    Parameters
    ----------
    t : string
        Input token to be cleansed.
    remchars : string, optional
        Compiled as a regex pattern. Characters matching are entirely removed from tokens.
        The default is "[`¬@^\|\"]".

    Returns
    -------
    string
        Cleansed input token (may be empty string).

    '''
    import re
    
    re_nonw  = re.compile("\W+")
    re_lead  = re.compile("^[\*\'#~&\(\)\[\]\{\}<>\.,_?!+=\\\\\/\-:;]+(.+)")
    re_trail = re.compile("(.+?)[\*#~&\(\)\[\]\{\}<>\.,_?!+=\\\\\/\-:;]+$")  # First group is lazy to get multiple trailing chars e.g. ")."
    re_del   = re.compile(remchars)      # Characters to be deleted altogether
    re_list  = re.compile("[0-9]{1,2}[\)\.:-]") # Occasional list entries like "1) A list" - omitted
    re_quote = re.compile("^([\"\'])(.+)\\1[\.!?;:,\)\}\]]?$") # Allow optional (specific) punctuation after closing quote
    re_pix   = re.compile("\d{1,4}px")
    
    re_quend = re.compile("^([^\'].*?)[\']+$")
    apos_ok = ["o'", "O'", "t'", "wi'"]  # Strings ending with single quote which are unchanged - otherwise only allowed if preceded by s
                                         # This breaks some dialect and various foreign inclusions (e.g. Xhosa, Klingon...) but is pretty good for standard Eng
    
    
    if re_nonw.fullmatch(t) or re_list.fullmatch(t): # List enumerators and fully non-word tokens omitted
            t = ''
    else:
        t = re.sub(re_del,'',t)       # Remove all backticks, carets pipes etc
        t = re.sub(re_quote,group2,t) # Remove surrounding paired quotes
        t = re.sub(re_lead,group1,t)  # Remove leading brackets, fullstops, hyphens, underscores, commas, slashes, colons, exclams
        t = re.sub(re_trail,group1,t) # Remove trailing things - similar to lead, plus #~
        
        # Specific handling for trailing single quote(s)
        if re_quend.fullmatch(t):
            # Allow exact token "o'", either case (Tam O' Shanter , will o' the wisp), "t'" (lower) and "wi'" (lower)
            if t in apos_ok: pass
            # Allow trailing ' otherwise only if singular and preceded by an s
            elif t[-2:] == "s'": pass
            else: t = re.sub(re_quend,group1,t)
            
        # n-gram count inflation caused by repetitions of e.g. '180px' (from Wiki image descriptions). Remove these.
        if re_pix.fullmatch(t):
            t = ''
        
    return t.strip()


def tup_to_str(tup, tagmark='|', unk='<unk>'):
    ''' Converts 2-ple to string, separated by tagmark. '''
    if tup[0] == unk:
        return tup[0]
    else:
        return tup[0]+tagmark+tup[1]
    
    
def yielder(s_out, asstr = False, tokenizer = None, unk='<unk>', tagger=None, tagmark = '|'):              
        if tokenizer == None:
            pass
        else:   # Intended use - apply an instance of NLTK's MWETokenizer to sentences
            s_out = tokenizer.tokenize(s_out)

        if tagger == None:
            pass
        else: # Tagger supplied - apply to sentence. Don't append tags to unk tokens.
            s_out = [ tup_to_str(tup,tagmark=tagmark, unk=unk) for tup in tagger.tag(s_out)]
            
        if asstr:
            return ' '.join(s_out)
        else:
            return s_out
            

def sent_gen(corpus, 
             remchars = "[`¬@^\|\"]",
             maxlen = 0,
             tokenizer = None, 
             vocab = None, unk = '<unk>', 
             tagger = None, tagmark = '|', 
             asstr = False):
    
    bad_sents = [ ['References'], ['Other', 'websites'], ['Related', 'pages'], ['Notes']]
    for sent in corpus.sents():
        s_out = []
        for t in sent:
            t = clean_token(t,remchars = remchars)
            if len(t):
                if vocab == None:
                    s_out.append(t)
                elif t in vocab:
                    s_out.append(t)
                else:
                    s_out.append(unk)
                         
        if len(s_out) and s_out not in bad_sents:  # Don't yield empty sentences or selected wiki-isms
            if maxlen <= 0:
                yield yielder(s_out, asstr = asstr, tokenizer = tokenizer, unk=unk, tagger=tagger, tagmark = tagmark)
            
            else:
                left= len(s_out)
                to_yield = []
                
                while left:
                    if left > 1.5*maxlen:
                        to_yield.append(yielder(s_out[0:maxlen], asstr = asstr, tokenizer = tokenizer, unk=unk, tagger=tagger, tagmark = tagmark))
                        s_out = s_out[maxlen:]
                        left = len(s_out)
                    else:
                        to_yield.append(yielder(s_out, asstr = asstr, tokenizer = tokenizer, unk=unk, tagger=tagger, tagmark = tagmark))
                        left = 0
                
                yield from to_yield
        
        
        
def word_gen(corpus, sent_mark = '|^|', remchars = "[`¬@^\|\"]", tokenizer = None, vocab = None, unk = '<unk>'):
    # Word generator depends on sentences, yielding tokens from the sentence.
    #  These are followed by a sentence marker - this is intended to ensure that n-gram counts can be filtered to
    #  avoid crossing sentence boundaries.
    for sent in sent_gen(corpus, remchars=remchars, tokenizer = tokenizer, vocab = vocab, unk = unk):
        for t in sent:
            yield t
        if len(sent_mark): yield sent_mark
              
                
def simp_word_gen(corpus, sent_mark = '|^|'):
    # Word generator depends on sentences, yielding tokens from the sentence.
    #  These are followed by a sentence marker - this is intended to ensure that n-gram counts can be filtered to
    #  avoid crossing sentence boundaries.
    for sent in corpus.sents():
        for t in sent:
            yield t
        if len(sent_mark): yield sent_mark
            
            
class Sent_Seq(object):
    def __init__(self, corpus, tokenizer=None, vocab = None, unk = '<unk>',
                remchars = "[`¬@^\|\"]", maxlen = 0, tagger = None, tagmark = '|', asstr = False):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.unk = unk
        self.remchars = remchars
        self.maxlen = maxlen
        self.tagger = tagger
        self.tagmark = tagmark
        self.asstr = asstr
 
    def __iter__(self):
        bad_sents = [ ['References'], ['Other', 'websites'], ['Related', 'pages'], ['Notes']]
        for sent in self.corpus.sents():
            s_out = []
            for t in sent:
                t = clean_token(t,remchars = self.remchars)
                if len(t):
                    if self.vocab == None:
                        s_out.append(t)
                    elif t in self.vocab:
                        s_out.append(t)
                    else:
                        s_out.append(self.unk)

            if len(s_out) and s_out not in bad_sents:  # Don't yield empty sentences or selected wiki-isms
                if self.maxlen <= 0:
                    yield yielder(s_out, asstr = self.asstr, tokenizer = self.tokenizer, unk=self.unk, tagger=self.tagger, tagmark = self.tagmark)

                else:
                    left= len(s_out)
                    to_yield = []

                    while left:
                        if left > 1.5*self.maxlen:
                            to_yield.append(yielder(s_out[0:self.maxlen], asstr = self.asstr, tokenizer = self.tokenizer, unk=self.unk, tagger=self.tagger, tagmark = self.tagmark))
                            s_out = s_out[self.maxlen:]
                            left = len(s_out)
                        else:
                            to_yield.append(yielder(s_out, asstr = self.asstr, tokenizer = self.tokenizer, unk=self.unk, tagger=self.tagger, tagmark = self.tagmark))
                            left = 0

                    yield from to_yield
                    
                    
class Simp_Sent_Seq(object):
    def __init__(self, corpus, tokenizer=None):
        self.corpus = corpus
        self.tokenizer = tokenizer

    def __iter__(self):
        for sent in self.corpus.sents():
            yield yielder(sent, asstr = False, tokenizer = self.tokenizer, tagger=None)
