
import execnet, remote_cooccurrence
from scipy import sparse
import numpy as np
import pickle

def dist_cooccurrence(vocab_dict, sentences, window_size = 10, mark = 2500, rem_return = None, specs=[('popen', 2)]):
    # v3 - all channels count their volume and return their results so far (and reset) periodically
    #  Should reduce memory usage in the remote channels by pulling their results back
    
    vocab_size = len(vocab_dict)
    
    if rem_return == None:
        rem_return = mark
    
    # Master matrix - ones returned by the exec channels are added to this one
    main_cooccurrence = sparse.csr_matrix((vocab_size,vocab_size), dtype=np.float64)
       
    group = execnet.Group()

    for spec, count in specs:
        for i in range(count):
            group.makegateway(spec)
            
    # Pickle dictionary for sending to remote channels
    vocab_dict_p = pickle.dumps(vocab_dict)
            
    mch = group.remote_exec(remote_cooccurrence)
    # Send parameters to each channel
    mch.send_each((vocab_dict_p, vocab_size, window_size, rem_return))
    
    queue = mch.make_receive_queue(endmarker = 'closed')
        
    terminated = 0
    _done = 0
    
    sent = []
    
    while 1:
        channel, item = queue.get()
        
        # Pickled matrix returned by remote on request
        if isinstance(item, bytes):
            main_cooccurrence += pickle.loads(item)
            print('Obtained response from channel {}'.format(channel))
            
            # That should be final return for this channel if we're out of sentences
            if sent == None:
                channel.send('terminate')
        
        # Remote channel indicates that it has terminated
        elif item == 'term':
            terminated += 1
            print('Channel {} terminated. {} so far.'.format(channel,terminated))
            # Channel terminated
            if terminated == len(mch):
                print('Everything terminated, exiting')
                # All results obtained, terminate loop
                break
            continue
        
        # Remote channel done processing, ready for more data
        elif item == 'sendmore':
            
            _done += 1
            if _done % mark == 0:
                print(' Processed: '+str(_done))
                
            sent = next(sentences, None)
        
            if sent == None: # No more sentences - request matrices from all remote processes
                print('Out of sentences - requesting returns')
                #mch.send_each('return')
                channel.send('return')
            else:
                channel.send(sent)
               
        # Remote channel closed - wait briefly then continue
        elif item == 'closed':
            continue
            
    group.terminate()
    
    return main_cooccurrence
            