import os

path = 'C:/Users/'+os.getlogin()+'/Google Drive/University/Dissertation'
datapath = 'C:/Users/'+os.getlogin()+'/Dissertation Data'

os.chdir(path+'/Code')


import pandas as pd
import numpy as np

import multiprocessing as mp

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import MWETokenizer, WhitespaceTokenizer

from glove import Corpus, Glove

   
# Import word and sentence generators
from generators import sent_gen, word_gen, Sent_Seq

from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from nltk.metrics import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    NgramAssocMeasures,
)

from nltk.metrics.spearman import (
    spearman_correlation,
    ranks_from_scores,
)

from nltk import FreqDist

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from batcher import batcher  # Custom module with logic for assigning n-grams to batches, avoiding overlap




# Flatten down to a single number
def cosim(x,y):
    return cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0]


def mwe_score(exp, model, stats_frame):
    # Combined token for MWE
    mwetoken = '+'.join(exp)

    # Stopwords - 1 if component is a stopword, 0 if present, -1 if simplex word missing from vocab, -2 if MWE missing
    sws = []
    # Component vectors
    cvs = []

    #  Neighbours in original & MWE-aware space
    oldn = []
    newn = []

    # List of individual word similarities (where present in the vocab)
    css = []

    # Empty array
    earr = np.empty(1000)
    earr[:] = np.nan

    # Check that combined token exists in the vocab. This protects against inflation of n-gram counts caused by repeats
    #  of the same token (e.g. in lists like https://simple.wikipedia.org/wiki/List_of_cities,_towns_and_villages_in_Fars_Province)
    if mwetoken in model.dictionary:

        mwv = model.word_vectors[model.dictionary[mwetoken]]

        for w in exp:
            if w in model.dictionary:
                cvs.append(model.word_vectors[model.dictionary[w]])

                oldn.append(model.most_similar(w, number=5))

                if w in stop:
                    sws.append(1)
                    css.append(np.nan)
                else:
                    sws.append(0)
                    css.append(cosim(model.word_vectors[model.dictionary[w]], mwv ))

            # If component is absent from vocab
            else:
                sws.append(-1)
                cvs.append(earr)
                css.append(np.nan)

                oldn.append([])

        #  Mean cosim
        if min(sws) >= 0:
            cs = np.nanmean(css)
        else:
            cs = np.nan

        newn = model.most_similar(mwetoken, number=5)

    # Combined token missing from vocab - mark with defaults
    else:
        sws = [-2]
        mwv = np.empty(400)
        mwv[:] = np.nan


    # Append to stats df
    return stats_frame.append({
        'ngram'  : exp,
        'stopwords' : sws,
        'mwe_vector' : mwv,
        'component_vectors' : cvs,
        'component_cosims'  : css,
        'cosine_sim'  : cs,
        'base_nearest': oldn,
        'mwe_nearest' : newn,
    }, ignore_index=True)


def mwe_score_par(args):
    #exp, model = args
    exp = args
    
    # Combined token for MWE
    mwetoken = '+'.join(exp)

    # Stopwords - 1 if component is a stopword, 0 if present, -1 if simplex word missing from vocab, -2 if MWE missing
    sws = []
    # Component vectors
    cvs = []

    #  Neighbours in original & MWE-aware space
    oldn = []
    newn = []

    # List of individual word similarities (where present in the vocab)
    css = []

    # Empty array
    earr = np.empty(1000)
    earr[:] = np.nan

    # Check that combined token exists in the vocab. This protects against inflation of n-gram counts caused by repeats
    #  of the same token (e.g. in lists like https://simple.wikipedia.org/wiki/List_of_cities,_towns_and_villages_in_Fars_Province)
    if mwetoken in batch_model.dictionary:

        mwv = batch_model.word_vectors[batch_model.dictionary[mwetoken]]

        for w in exp:
            if w in batch_model.dictionary:
                cvs.append(batch_model.word_vectors[batch_model.dictionary[w]])

                oldn.append(batch_model.most_similar(w, number=5))

                if w in stop:
                    sws.append(1)
                    css.append(np.nan)
                else:
                    sws.append(0)
                    css.append(cosim(batch_model.word_vectors[batch_model.dictionary[w]], mwv ))

            # If component is absent from vocab
            else:
                sws.append(-1)
                cvs.append(earr)
                css.append(np.nan)

                oldn.append([])

        #  Mean cosim
        if min(sws) >= 0:
            cs = np.nanmean(css)
        else:
            cs = np.nan

        newn = batch_model.most_similar(mwetoken, number=5)

    # Combined token missing from vocab - mark with defaults
    else:
        sws = [-2]
        mwv = np.empty(400)
        mwv[:] = np.nan


    # Return stats df
    return pd.DataFrame.from_dict({
        'ngram'  : [exp],
        'stopwords' : [sws],
        'mwe_vector' : [mwv],
        'component_vectors' : [cvs],
        'component_cosims'  : [css],
        'cosine_sim'  : [cs],
        'base_nearest': [oldn],
        'mwe_nearest' : [newn],
    })

def main():
    
    simp = PlaintextCorpusReader(datapath+'/Corpora/wiki/simple_20200601/','simple_sample.txt',
                                word_tokenizer = WhitespaceTokenizer()
                                )
    
    # Collate n-grams
    scorer = NgramAssocMeasures.poisson_stirling

    tri_cf = TrigramCollocationFinder.from_words(word_gen(simp))
    tri_cf.apply_word_filter(lambda w: w in ('|^|'))  # Filter out associations with sentence boundary marker
    
    bi_cf = tri_cf.bigram_finder()                   # Make bigram finder from trigram, don't need to count again
    bi_cf.apply_word_filter(lambda w: w in ('|^|'))  # Filter out associations with sentence boundary marker
    
    bi_dict = {}
    tri_dict = {}

    for bigram in bi_cf.score_ngrams(scorer):    
        bi_dict[bigram[0]] = [bigram[0], bi_cf.ngram_fd[bigram[0]], bigram[-1]]

    for trigram in tri_cf.score_ngrams(scorer):    
        tri_dict[trigram[0]] = [trigram[0], tri_cf.ngram_fd[trigram[0]], trigram[-1]]

    bigram_df = pd.DataFrame.from_dict(bi_dict, orient='index',
                           columns=['ngram', 'freq', 'poisson'])

    trigram_df = pd.DataFrame.from_dict(tri_dict, orient='index',
                           columns=['ngram', 'freq', 'poisson'])
    
    ngram_df = bigram_df.append(trigram_df).sort_values('poisson', ascending=False).reset_index(drop=True)
    ngram_df['len'] = ngram_df.ngram.apply(len)
    
    
    ## Clean up
    del tri_cf, bi_cf, bigram_df, trigram_df, bi_dict, tri_dict
    
    
    # Stopwords from corpus - 50 most frequent
    fdist = FreqDist(word_gen(simp, sent_mark=''))

    stop = set( word for word, f in fdist.most_common(20))
    
    
    # Minimum n-gram frequency, no. of n-gram candidates to evaluate
    min_freq = 10
    eval_count = 150000
    
    
    # Duplicate entries appearing for some reason. Removing here
    ngram_df2 = ngram_df[ngram_df.freq >= min_freq].drop_duplicates().reset_index(drop=True)

    ngram_eval = ngram_df2[0:eval_count]

    # Clean up
    del ngram_df, ngram_df2
    
    # Assign n-grams to batches
    batches, batch_count = batcher(ngram_eval.ngram, stopwords=stop, max_batches = 15)
    
    # Should be able to add batch information using df.map() but am encountering errors apparently relating
    #  to indexing - workaround (though slower).
    ngb_cols = ["ngram", "batch"]
    rows = []

    for ng in ngram_eval['ngram']:
        rows.append({"ngram" : ng,
                    "batch" : batches[ng]})

    ng_batch = pd.DataFrame(rows, columns = ngb_cols)

    ngram_eval = ngram_eval.merge(ng_batch, on='ngram')
    
    del ng_batch
    
    
    # Process batches
    batch_dfs = {}

    for bb in range(batch_count):
        print('Processing batch {} of {}'.format(bb+1,batch_count))
        # Subset DataFrame
        batch_dfs[bb] = ngram_eval[ngram_eval.batch == bb+1].reset_index(drop=True)

        # Initialise MWETokenizer
        batch_token_mwe = MWETokenizer(list(batch_dfs[bb].ngram) , separator='+')

        # Build model
        simp_corp = Corpus()

        sents_mwe = Sent_Seq(simp, batch_token_mwe)
        simp_corp.fit( sents_mwe , window = 10)

        batch_model = Glove(no_components = 300, 
                 learning_rate = 0.05)

        batch_model.fit(simp_corp.matrix, 
              epochs=50,
              no_threads=16,
              verbose=False)

        batch_model.add_dictionary(simp_corp.dictionary)

        # Save model
        batch_model.save(datapath+'/Models/2 GloVe/simple_batch{}.model'.format(bb+1))
        # Reload looks like    new_model = Glove.load('glove.model')

        # For each MWE, evaluate stats. Record vectors (in case we want to calculate different metrics later).
        # Parallelized version
        with mp.Pool() as pool:
            statslist = pool.map( mwe_score_par , batch_dfs[bb].ngram )

        # statslist = [mwe_score_par((ng, batch_model)) for ng in batch_dfs[bb].ngram]

        statsf = pd.concat(statslist)

        #  Join back onto DataFrame
        batch_dfs[bb] = batch_dfs[bb].merge(statsf, on='ngram')
        
        
    # Merge dataframes, sort by compositionality metric, export
    # Also want the default batches with batch no < 0
    all_batches = ngram_eval[ngram_eval.batch < 0]

    for d in range(batch_count):
        all_batches = all_batches.append(batch_dfs[d])

    all_batches = all_batches.sort_values('cosine_sim')
    all_batches = all_batches.reset_index(drop=True)

    all_batches.to_csv(datapath+'/Models/2 GloVe/Results/simple_output_001.csv', index=False)
    
    
if __name__ == '__main__':
    mp.freeze_support()
    
    main()
    
