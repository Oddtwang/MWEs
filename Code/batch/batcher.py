def tup_overlap(tupA, tupB):
    ''' Identifies overlap between two tuples '''
    for i in range(min(len(tupA),len(tupB))):
        if tupA[-(i+1):] == tupB[:i+1] or tupB[-(i+1):] == tupA[:i+1]:
            return True
    return False


def batcher(tuples, max_batches = -1, stopwords=[], invert=False):
    '''
    By default, returns a dictionary of the processed tuples and their assigned batches.
    If invert=True, returns dict of batch numbers and a list of tuples assigned to each.
    Also returns the total number of batches.
    
    If max_batches > 0, anything which would be assigned to a higher batch number is assigned -1.
    Tuples comprised entirely of entries in stopwords are assigned to batch -2.
    '''
    # Inititalise dict for batch numbers and their contents
    batch_dict = { 1 : [] }
    if max_batches > 0 : batch_dict[-1] = []
    if len(stopwords)  : batch_dict[-2] = []
        
    class ContinueB(Exception):
        pass

    class ContinueT(Exception):
        pass

    continue_b = ContinueB()
    continue_t = ContinueT()
    
    
    for t in tuples:
        #print("\nProcessing:", t)
        
        # Check if all words in t are in the stopwords list - if so, assign to batch -2
        if len(stopwords) and all( w in stopwords for w in t):
            stop_items = batch_dict[-2]
            stop_items.append(t)
            batch_dict[-2] = stop_items
            
        else:
            try:
                for b in [x for x in batch_dict.keys() if x>0]:  # Iterate over existing batches, excluding catchall batch -1
                    b_items = batch_dict[b] 
                    try:
                        for k in b_items:         # Items already assigned to this batch
                            if tup_overlap(k,t):   # Overlap (incl. subsets) - conflict
                                #print("Conflicts with batch", str(b))
                                raise continue_b
    
                        # No conflict with this batch - assign to it, move on to next tuple
                        #print("Assigning to batch", str(b))
                        b_items.append(t)
                        batch_dict[b] = b_items
                        raise continue_t
    
                    except ContinueB:
                        continue
                        
                # Got through all the existing batches without assigning
                #print("Unable to assign to existing batch")
                bmax = max(batch_dict.keys())
                
                # Batch limit reached - assign tuple to batch -1
                #print("Max batches: ", str(max_batches))
                #print("Assigned batches: ", str(bmax))
                if max_batches > 0 and bmax >= max_batches:
                    def_items = batch_dict[-1]
                    def_items.append(t)
                    batch_dict[-1] = def_items
                
                # Create new batch and assign tuple to it
                else:
                    batch_dict[bmax+1] = [t]    
            
            except ContinueT:
                continue
            
    if invert:
        return batch_dict, max(batch_dict.keys())
    
    else:
        batches = {}
        for k, v in batch_dict.items():
            for m in v:
                batches[m] = k
        return batches, max(batch_dict.keys())