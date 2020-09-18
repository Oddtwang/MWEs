
import itertools, execnet, r_remote_bigram_score, r_remote_trigram_score

from redis import Redis
from redisprob import RedisHashFreqDist


def dist_score_bigrams(bigrams,  fdnames, bgn, odname, host='localhost', total = 0, min_freq = 0, mark = 1000, specs=[('popen', 2)]):
    gateways = []
    channels = []

    for spec, count in specs:
        for i in range(count):
            gw = execnet.makegateway(spec)
            gateways.append(gw)
            
            channel = gw.remote_exec(r_remote_bigram_score)
            channel.send((host, fdnames, bgn, odname, min_freq))
            channels.append(channel)

    cyc = itertools.cycle(channels)

    if total <= 0:
        total = len(bigrams)
    
    print('Bigrams to process: '+str(total))
    
    _done = 0
    
    for bigram in bigrams:
        channel = next(cyc)
        channel.send(bigram)
        _done += 1
        if _done % mark == 0:
            print(' Processed: '+str(_done)+' / '+str(total))

    for channel in channels:
        channel.send('done')
        assert 'done' == channel.receive()
        channel.waitclose(5)

    for gateway in gateways:
        gateway.exit()
        
        
def dist_score_trigrams(trigrams, fdnames, tgn, odname, host='localhost', total = 0, min_freq = 0, mark = 1000, specs=[('popen', 2)]):
    gateways = []
    channels = []

    for spec, count in specs:
        for i in range(count):
            gw = execnet.makegateway(spec)
            gateways.append(gw)
            
            channel = gw.remote_exec(r_remote_trigram_score)
            channel.send((host, fdnames, tgn, odname, min_freq))
            channels.append(channel)

    cyc = itertools.cycle(channels)
    
    if total <= 0:
        total = len(trigrams)
    
    print('Trigrams to process: '+str(total))
    
    _done = 0
    for trigram in trigrams:
        channel = next(cyc)
        channel.send(trigram)
        _done += 1
        if _done % mark == 0:
            print(' Processed: '+str(_done)+' / '+str(total))

    for channel in channels:
        channel.send('done')
        assert 'done' == channel.receive()
        channel.waitclose(5)

    for gateway in gateways:
        gateway.exit()
    