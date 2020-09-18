
import itertools, execnet, r_remote_ngcount

from redis import Redis
from redisprob import RedisHashFreqDist



def dist_count_ngrams(sentences, fdnames, minl, maxl, host='localhost', specs=[('popen', 2)]):
    gateways = []
    channels = []

    for spec, count in specs:
        for i in range(count):
            gw = execnet.makegateway(spec)
            gateways.append(gw)
            channel = gw.remote_exec(r_remote_ngcount)
            channel.send((host, fdnames, minl, maxl))
            channels.append(channel)

    cyc = itertools.cycle(channels)

    for sentence in sentences:
        channel = next(cyc)
        channel.send(sentence)

    for channel in channels:
        channel.send('done')
        assert 'done' == channel.receive()
        channel.waitclose(5)

    for gateway in gateways:
        gateway.exit()

    # Return tuple of total counts
    r = Redis(host)
 
    _Ns = []
    
    for fdn in fdnames:
        fd = RedisHashFreqDist(r, fdn)
        _Ns.append(fd.N())
    
    return tuple(_Ns)