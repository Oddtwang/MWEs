
from redis import Redis
from redisprob import RedisHashFreqDist

from nltk.util import everygrams

if __name__ == '__channelexec__':
    host, fdnames, minl, maxl  = channel.receive()
    r = Redis(host)
    
    ug_name, big_name, trg_name = fdnames
    
    ug = RedisHashFreqDist(r, ug_name)
    big = RedisHashFreqDist(r, big_name)
    trg = RedisHashFreqDist(r, trg_name)
    

    for data in channel:
        if data == 'done':
            channel.send('done')
            break
        
        # everygrams(x,1,3) returns all uni-, bi- and tri-grams from list (sentence) x.
        # for each, update corresponding hashed frequency distribution object
        for gram in everygrams(data,minl,maxl):
            _gl = len(gram)
            if _gl == 1:
                ug[gram] += 1
            if _gl == 2:
                big[gram] += 1
            if _gl == 3:
                trg[gram] += 1
            else:
                pass