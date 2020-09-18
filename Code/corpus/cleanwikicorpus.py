
import sys
sys.path.append('C:\\Anaconda\\Lib\\site-packages\\gensim\\corpora\\')

from nltk.tokenize import  WhitespaceTokenizer
import nltk.data
import re

from wikicorpus import *


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


reA = re.compile("\'{2,}", re.UNICODE) # Multiple quotes
reB = re.compile("={2,}", re.UNICODE)  # Multiple equals
reB2 = re.compile("={2,}$", re.UNICODE)  # Multiple equals, at the end of a token

reC = re.compile("^(\w{0,}[.!?;:\"\(\)])'{2,}$") # Sentence ender followed by double quotes - want to retain only the former

reD = re.compile("\*+\s*(\'{0,5}[A-Z])", re.UNICODE) # Asterisk(s) followed by capital (with optional space) - assume bullet point, sub * for .

def sent_end_rep(matchobj):
    return matchobj.group(1)

def bullet_rep(matchobj):
    return '. '+matchobj.group(1)


def tokenize(content, token_min_len = 1, token_max_len = 15, lower=True):
    #used to override original method in wikicorpus.py
    outlist = []
    # Sentences
    for sent in sent_detector.tokenize(re.sub(reD,bullet_rep,content)): # Asterisk followed by capital - assume bullet marker, swap * for .
        for t in WhitespaceTokenizer().tokenize(sent):
            if len(t) >= token_min_len and len(t) <= token_max_len and not t.startswith('_'):
                if re.fullmatch(reA,t):  # Omit markup tokens of multiple quotes
                    continue
                elif re.fullmatch(reB,t): # Multiple equals (heading markup) -> newline
                    t = '\n'
                elif re.fullmatch(reC,t):
                    t = re.sub(reC,sent_end_rep,t) # Retain sentence enders followed by double quotes
                    
                t = re.sub(reA,'',t) # Remove blocks of multiple quotes (from markup)
                t = re.sub(reB2,'\n',t) # Multiple equals at end of a token should be a newline (to keep headings as sentences)
                t = re.sub(reB,'',t) # Remove blocks of multiple equals (from markup)
                    
                if lower:
                    outlist.append(utils.to_unicode(t.lower()))
                else:
                    outlist.append(utils.to_unicode(t))
                    
        outlist.append('\n')
    return outlist



def process_article(args, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
   # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text, token_min_len, token_max_len, lower)
    return result, title, pageid


def _process_article(args):
    """Same as :func:`~gensim.corpora.wikicorpus.process_article`, but with args in list format.

    Parameters
    ----------
    args : [(str, bool, str, int), (function, int, int, bool)]
        First element - same as `args` from :func:`~gensim.corpora.wikicorpus.process_article`,
        second element is tokenizer function, token minimal length, token maximal length, lowercase flag.

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    Warnings
    --------
    Should not be called explicitly. Use :func:`~gensim.corpora.wikicorpus.process_article` instead.

    """
    tokenizer_func, token_min_len, token_max_len, lower = args[-1]
    args = args[:-1]

    return process_article(
        args, token_min_len=token_min_len,
        token_max_len=token_max_len, lower=lower
    )


class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), lower=True, token_min_len=1, token_max_len=15, dictionary=None, filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=50, filter_articles=None):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces, tokenizer_func, article_min_tokens, token_min_len, token_max_len, lower, filter_articles)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = \
            ((text, self.lemmatize, title, pageid, tokenization_params)
             for title, text, pageid
             in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces, self.filter_articles))
        pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)
        
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(_process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                positions += len(tokens)
                if self.metadata:
                    yield (tokens, (pageid, title))
                else:
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length