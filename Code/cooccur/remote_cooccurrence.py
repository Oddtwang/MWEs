
from scipy import sparse
import numpy as np
import pickle
from gensim.corpora import Dictionary

if __name__ == '__channelexec__':
    """
    Based on https://github.com/hans/glove.py/
    This version yields a (small) sparse matrix for a single sentence, in scipy's CSR format for efficient
     combination with others by summing - this will allow for distributed creation of co-occurrence matrix.
    
    vocab_dict is a gensim Dictionary mapping the vocabulary of the corpus to IDs.
    
    This version maintains a matrix in the remote channel throughout, and sends it back when 
    it is requested by the main process
    """
    vocab_dict_p, vocab_size, window_size, rem_return = channel.receive()
    
    vocab_dict = pickle.loads(vocab_dict_p)
    
    # Collect cooccurrences internally as a sparse matrix
    my_cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)
    my_count = 0
    
    channel.send('sendmore')
    
    for data in channel:
        if data == 'terminate':
            channel.send('term')
            break
        
        # Master has requested matrix back
        elif data == 'return':
            # Convert to CSR format for efficient combination with matrices from other sentences
            channel.send(pickle.dumps(my_cooccurrences.tocsr()))
            # Reset matrix (enables variation in which master requests matrix periodically
            #  - this ought to reduce memory usage in remote channels)
            my_cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)
   
        else:
            my_count += 1
            # Channel data is a sentence - list of tokens
            token_ids = [vocab_dict.token2id[word] for word in data]
        
            for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
                context_ids = token_ids[max(0, center_i - window_size) : center_i]
                contexts_len = len(context_ids)
        
                for left_i, left_id in enumerate(context_ids):
                    # Distance from center word
                    distance = contexts_len - left_i
                    #distance = 1
        
                    # Weight by inverse of distance between words
                    increment = 1.0 / float(distance)
        
                    # Build co-occurrence matrix symmetrically (pretend we
                    # are calculating right contexts as well)
                    my_cooccurrences[center_id, left_id] += increment
                    my_cooccurrences[left_id, center_id] += increment
                    
            if my_count >= rem_return:
                channel.send(pickle.dumps(my_cooccurrences.tocsr()))
                # Reset matrix
                my_cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)
                my_count = 0
                
            # Ask master for more data
            channel.send('sendmore')
                
        
        
     


