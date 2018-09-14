import math
import collections
import numpy as np
import os

from context2vec.common.defs import Toks, SENT_COUNTS_FILENAME, WORD_COUNTS_FILENAME
from collections import OrderedDict


def read_batch(f, batchsize, word2index, tags2index):        
    batch = []
    batchtags = []
    os.path.basename(f.name)
    sentlen  =  int(os.path.basename(f.name).split(".")[1])  
   
    while len(batch) < batchsize:
        line = f.readline()
        if not line: break
#        if len(line) != sentlen: continue
        sent_tags = line.split()
        sent_tags_inds = []
        error = False
        for tag in sent_tags:
            split = tag.split("_")
            if len(split) == 2:
                tag = tag.split("_")[1]
                if tag in tags2index:
                    ind = list(tags2index.keys()).index(tag)
                else:
                    ind = list(tags2index.keys()).index('<UNK>')
                sent_tags_inds.append(ind)
            else:
                 error = True
                 break
        if error == True:
            continue
            
              
        sent_words = line.strip().lower().split()
        assert(len(sent_words) > 1)
        sent_inds = []
        for word in sent_words:
            word = word.split("_")[0]
            if word in word2index:
                ind = word2index[word]
            else:
                ind = word2index['<UNK>']
            sent_inds.append(ind)
            
          
        if len(sent_tags_inds) != sentlen  or len(sent_inds)  != sentlen:
            print(f.name)
            continue
        batchtags.append(sent_tags_inds)  
        batch.append(sent_inds)
            
    return batch,batchtags


class SentenceReaderDir(object):
    '''
    Reads a batch of sentences at a time from a corpus directory in random order.
    Assumes that the sentences are split into different files in the directory according to their word lengths.
    '''
    
    sent_counts_filename = SENT_COUNTS_FILENAME
    word_counts_filename = WORD_COUNTS_FILENAME

    def __init__(self, path, trimfreq, batchsize):
        '''
        Initialize reader.
        :param path: input directory
        :param trimfreq: treat all words with lower frequency than trimfreq as unknown words
        :param batchsize: the size of the minibatch that will be read in every iteration
        '''
        self.path = path
        self.batchsize = batchsize
        self.trimmed_word2count, self.word2index, self.index2word , self.tag2index = self.read_and_trim_vocab(trimfreq)
        self.total_words = sum(self.trimmed_word2count.values())
        self.fds = []
        
    def open(self):
        self.fds = []
        with open(self.path+'/'+self.sent_counts_filename) as f:
            for line in f:
                [filename, count] = line.strip().split()
                batches = int(math.ceil(float(count) / self.batchsize))
                fd = open(self.path+'/'+filename, 'r',errors='replace')
                self.fds = self.fds + [fd]*batches
        np.random.seed(1034)
        np.random.shuffle(self.fds)
    

    def close(self):
        fds_set = set(self.fds)
        for f in fds_set:
            f.close()
                        
            
    def read_and_trim_vocab(self, trimfreq):
        word2count = collections.Counter()
        with open(self.path+'/'+self.word_counts_filename) as f:
            for line in f:
                [word, count] = line.strip().lower().split()
                word2count[word] = int(count)
    
        trimmed_word2count = collections.Counter()
        index2word = {Toks.UNK:'<UNK>', Toks.BOS:'<BOS>', Toks.EOS:'<EOS>'}
        word2index = {'<UNK>':Toks.UNK, '<BOS>':Toks.BOS, '<EOS>':Toks.EOS}
        unknown_counts = 0
        for word, count in word2count.items():
            if count >= trimfreq and word.lower() != '<unk>' and word.lower() != '<rw>':    
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
                trimmed_word2count[ind] = count
            else:
                unknown_counts += count
        trimmed_word2count[word2index['<BOS>']] = 0
        trimmed_word2count[word2index['<EOS>']] = 0
        trimmed_word2count[word2index['<UNK>']] = unknown_counts
        

        
        tag2index =  OrderedDict()
        tag2index['<UNK>'] =0

        with open(self.path+'/'+self.sent_counts_filename) as f:
            for line in f:
                [filename, count] = line.strip().split()                
                fd = open(self.path+'/'+filename, 'r',errors='replace')
              
                for line in fd:
                    sent_tags = line.split()
                    
                    for tag in sent_tags:
                        split = tag.split("_")
                        if len(split) == 2:
                            tag = split[1]    
                            tag2index[tag]=0
                        else:
                            print(tag)
                            print(filename)
        return trimmed_word2count, word2index, index2word , tag2index

       
    def next_batch(self):                
        for fd in self.fds:
            batch,batchtags = read_batch(fd, self.batchsize, self.word2index , self.tag2index)
            yield (batch,batchtags)
 
 
 
if __name__ == '__main__':
    import sys   
    reader = SentenceReaderDir(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    
    for i in range(2):
        print('epoc', i)
        reader.open()
        i = 0
        j = 0
        for batch in reader.next_batch():
            if i < 3:
                print(batch)
                print()
            i += 1
            j += len(batch)
        print('batches', i)
        print('sents', j)
        reader.close()
                     
