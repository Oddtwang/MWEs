
import os

from redis import Redis

import datetime
start = datetime.datetime.now()


print('Script execution started at: ', start)


path = 'C:/Users/'+os.getlogin()+'/Google Drive/University/Dissertation'
datapath = 'E:/Dissertation Data'

os.chdir(path+'/Code')

logfilename = './Logs/w10p_score_ngrams.txt'

with open(logfilename, 'w') as logfile:
    logfile.write('Script execution started at: ')
    logfile.write(str(start))
    logfile.write('\n')

from r_dist_ngram_scores import dist_score_bigrams, dist_score_trigrams
from rediscollections import keys_to_tuples
from redisprob import RedisHashFreqDist

r = Redis('localhost')

bg = RedisHashFreqDist(r, 'w10p:w10p_bg')
tg = RedisHashFreqDist(r, 'w10p:w10p_tg')

min_freq = 20



bigrams = keys_to_tuples(bg)


#from nltk.metrics import (
#    BigramAssocMeasures,
#    TrigramAssocMeasures,
#    NgramAssocMeasures,
#)

#score_bi= BigramAssocMeasures.poisson_stirling
#score_tri= TrigramAssocMeasures.poisson_stirling

# Get total number of bigrams - only want to make this call once rather than having 
#  each channel repeat it for each bigram
_bgn = bg.N()

dist_score_bigrams(bigrams, ('w10p:w10p_ug', 'w10p:w10p_bg'), _bgn, 'w10p:w10p_poisson', 
                   total = len(bg), min_freq = min_freq, mark = 200000, specs=[('popen', 6)])


bi_done = datetime.datetime.now()
bi_time = bi_done - start

with open(logfilename, 'a') as logfile:
    logfile.write('Bigram scoring completed at: ')
    logfile.write(str(bi_done))
    logfile.write('\n')
    logfile.write('Bigram scoring time: ')
    logfile.write(str(bi_time))
    logfile.write('\n')
    
trigrams = keys_to_tuples(tg)

# Get total number of bigrams - only want to make this call once rather than having 
#  each channel repeat it for each bigram
_tgn = tg.N()
    
dist_score_trigrams(trigrams, ('w10p:w10p_ug', 'w10p:w10p_tg'), _tgn, 'w10p:w10p_poisson', 
                    total = len(tg), min_freq = min_freq, mark = 200000, specs=[('popen', 6)])


from rediscollections import RedisOrderedDict

score_dict = RedisOrderedDict(r, 'w10p:w10p_poisson')
print(len(score_dict))


end = datetime.datetime.now()
total_time = end - start
trigram_time = end - bi_done

with open(logfilename, 'a') as logfile:
    logfile.write('Script execution completed at: ')
    logfile.write(str(end))
    logfile.write('\n')
    logfile.write('Trigram scoring time: ')
    logfile.write(str(trigram_time))
    logfile.write('\n')
    logfile.write('Total execution time: ')
    logfile.write(str(total_time))
    logfile.write('\n')
    logfile.write('ngrams scored: ')
    logfile.write(str(len(score_dict)))
    logfile.write('\n')
    
print("Script execution ended at: ", end)
print("Script totally ran for: ", total_time)

