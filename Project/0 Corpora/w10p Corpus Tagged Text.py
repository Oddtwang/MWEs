

import os

path = 'C:/Users/'+os.getlogin()+'/Google Drive/University/Dissertation'
datapath = 'C:/Users/'+os.getlogin()+'/Dissertation Data'
#datapath = 'E:/Dissertation Data'

os.chdir(path+'/Code')

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer

from generators import sent_gen, tup_to_str
from sner import POSClient


# Server command used
#java -Xmx12g -cp stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTaggerServer -port 9198 -model models/english-left3words-distsim.tagger -annotators "tokenize,ssplit,pos"

tagger = POSClient(host='localhost', port=9198)

w10p = PlaintextCorpusReader(datapath+'/Corpora/wiki/enwiki_20200520/','enwiki_20200520_10pc.txt',
                            word_tokenizer = WhitespaceTokenizer()
                            )


output = open(datapath+'/Corpora/wiki/enwiki_20200520/enwiki_20200520_10pc_tagged.txt', 'w', encoding='utf-8')

i = 0
for sent in sent_gen(w10p, asstr = True, maxlen = 70):
    sent = [ tup_to_str(tup) for tup in tagger.tag(sent)]
    output.write(bytes(' '.join(sent), 'utf-8').decode('utf-8')+' \n')
    i = i + 1
    if (i % 100000 == 0):
        print('Processed ' + str(i) + ' sentences')
        
output.close()
print('Processing complete!')