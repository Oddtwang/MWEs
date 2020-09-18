# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:56:51 2020

@author: tom
"""

import os

path = 'C:/Users/'+os.getlogin()+'/Google Drive/University/Dissertation'
datapath = 'C:/Users/'+os.getlogin()+'/Dissertation Data'

os.chdir(path+'/Code')

java_path = "C:/Program Files/Java/jdk-13.0.2/bin/java.exe"
os.environ['JAVAHOME'] = java_path

from nltk.tag import StanfordPOSTagger
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer

from generators import sent_gen, tup_to_str
from sner import POSClient

stanford_dir = datapath+"/stanford-postagger-full-2020-08-06"
modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
jarfile=stanford_dir+"/stanford-postagger.jar"

#tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)
#tagger.java_options='-mx15360m'


tagger = POSClient(host='localhost', port=9198)

samp = PlaintextCorpusReader(datapath+'/Corpora/wiki/simple_20200601/','simple_sample.txt',
                            word_tokenizer = WhitespaceTokenizer()
                            )


output = open(datapath+'/Corpora/wiki/simple_20200601/simple_sample_tagged_3w.txt', 'w', encoding='utf-8')

i = 0
for sent in sent_gen(samp, asstr = True):
    sent = [ tup_to_str(tup) for tup in tagger.tag(sent)]
    output.write(bytes(' '.join(sent), 'utf-8').decode('utf-8')+' \n')
    i = i + 1
    if (i % 100000 == 0):
        print('Processed ' + str(i) + ' sentences')
        
output.close()
print('Processing complete!')