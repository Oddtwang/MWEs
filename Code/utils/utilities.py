import re
from sklearn.metrics.pairwise import cosine_similarity

def cosim(x,y):
    ''' Cosine similarity, as a single number '''
    return cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0]


def tupallmatch(tupin, pattern):
    ''' Check whether all elements in a tuple match regex pattern '''
    return all(pattern.match(w) for w in tupin)

def tupmatchlist(tupin, patterns):
    ''' Check if all elements of a tuple match any of a list of regex patterns '''
    for w,p in zip(tupin, patterns):
        if p.match(w): pass
        else: return False
    return True

pipematcher = re.compile("(.*)\|")
pipematch2 = re.compile("\|([A-Z]{2,4}\$?|[\$\:,\.\"]|``|\-LRB\-?|\-RRB\-?)")

def g1(matchobj):
    return matchobj.group(1)

def tup_matcher(tupstr, pattern=pipematcher):
    ot = []
    for w in tupstr:
        if w == '|HYPH': w = '-|HYPH'
        if re.match(pattern, w):
            ot.append(g1(pattern.match(w)))
    return tuple(ot)


def tup_rem(tupstr, pattern=pipematch2):
    ot = []
    for w in tupstr:
        if w == '|HYPH': w = '-|HYPH'
        ot.append(re.sub(pattern,'',w))
    return tuple(ot)