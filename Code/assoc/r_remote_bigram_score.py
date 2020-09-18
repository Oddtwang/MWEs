
from redis import Redis
from redisprob import RedisHashFreqDist
from rediscollections import RedisOrderedDict

from nltk.metrics import BigramAssocMeasures

if __name__ == '__channelexec__':
    host, fdnames, bgn, od_name, min_freq  = channel.receive()
    score_fn = BigramAssocMeasures.poisson_stirling
    
    r = Redis(host)
    
    ug_name, big_name = fdnames
    
    ug = RedisHashFreqDist(r, ug_name)
    big = RedisHashFreqDist(r, big_name)

    od = RedisOrderedDict(r, od_name)
    
    # Assume total # of bigrams is passed in - if not, calculate
    if bgn <= 0 :
        bgn = big.N()

    for data in channel:
        if data == 'done':
            channel.send('done')
            break
        
        # apply scoring function to bigram, store result in ordered dictionary
        # data received is bigram tuple
        # Apply minimum frequency to calc & record score
        _ngcount = big[data]
        if _ngcount >= min_freq:
            w1 = (data[0],)
            w2 = (data[1],)
        
            od[data] = score_fn( _ngcount , ( ug[w1] , ug[w2] ), bgn )