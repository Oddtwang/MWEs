

from redis import Redis
from redisprob import RedisHashFreqDist
from rediscollections import RedisOrderedDict

from nltk.metrics import TrigramAssocMeasures

if __name__ == '__channelexec__':
    host, fdnames, tgn, od_name, min_freq  = channel.receive()
    score_fn = TrigramAssocMeasures.poisson_stirling
    
    r = Redis(host)
    
    ug_name, trg_name = fdnames
    
    ug = RedisHashFreqDist(r, ug_name)
    trg = RedisHashFreqDist(r, trg_name)

    od = RedisOrderedDict(r, od_name)
    
    # Assume total # of trigrams is passed in - if not, calculate
    if tgn <= 0 :
        tgn = trg.N()

    for data in channel:
        if data == 'done':
            channel.send('done')
            break
        
        # apply scoring function to trigram, store result in ordered dictionary
        # data received is trigram tuple
        # Apply minimum frequency to calc & record score
        _ngcount = trg[data]
        if _ngcount >= min_freq:
            w1 = (data[0],)
            w2 = (data[1],)
            w3 = (data[2],)
            
            od[data] = score_fn(_ngcount , ( ug[w1] , ug[w2], ug[w3] ) , tgn )