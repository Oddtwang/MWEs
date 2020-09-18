
import os

path = 'C:/Users/'+os.getlogin()+'/Google Drive/University/Dissertation'
datapath = 'E:/Dissertation Data'

os.chdir(path+'/Code')

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer

from generators import sent_gen

import redis

import os

import datetime
start = datetime.datetime.now()
print("script execution stared at:", start)




w10p = PlaintextCorpusReader(datapath+'/Corpora/wiki/enwiki_20200520/','enwiki_20200520_10pc.txt',
                            word_tokenizer = WhitespaceTokenizer()
                            )

from r_dist_count import dist_count_ngrams

counts = dist_count_ngrams(sent_gen(w10p), ('w10p:w10p_ug', 'w10p:w10p_bg', 'w10p:w10p_tg'), 1, 3, 
                           specs=[('popen', 8)])

print(counts)

end = datetime.datetime.now()
print("Script execution ended at:", end)
total_time = end - start
print("Script totally ran for :", total_time)

