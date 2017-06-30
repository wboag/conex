

import sys
import numpy as np
import os


def load_word2vec_binary(fname, verbose=1, dev=False):
       """
       Loads 300x1 word vecs from Google (Mikolov) word2vec
       """
       word_vecs = {}
       if verbose:
           print 'loading word2vec'
       with open(fname, "rb") as f:
           header = f.readline()
           vocab_size, layer1_size = map(int, header.split())
           binary_len = np.dtype('float32').itemsize * layer1_size
           for line in xrange(vocab_size):
               # short circuit (when we just care about pipeline, not actually using this for tests)
               if dev:
                   if line >= 500:
                       break
               # display how long it takes?
               if verbose:
                   if line % (vocab_size/40) == 0:
                       print '%6.2f %%' % (100*float(line)/vocab_size)
               word = []
               while True:
                   ch = f.read(1)
                   if ch == ' ':
                       word = "".join(word)
                       break
                   if ch != '\n':
                       word.append(ch)
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
       return word_vecs



def load_glove_txt(fname, verbose=1, dev=False):
       """
       Loads 300x1 word vecs from GloVe vectors
       """
       word_vecs = {}
       if verbose:
           print 'loading glove'
       with open(fname, "r") as f:
           lines = f.readlines()
           N = len(lines)
           for i,line in enumerate(lines):
               # short circuit (when we just care about pipeline, not actually using this for tests)
               if dev:
                   if i >= 500:
                       break
               # display how long it takes?
               if verbose:
                   if i % (N/40) == 0:
                       print '%6.2f %%' % (100*float(i)/N)
               toks = line.strip().split()
               word = toks[0]
               vec = np.array(map(float,toks[1:]))
               word_vecs[word] = vec
       return word_vecs



def load_embeddings(vec_file, verbose=1, dev=False):
    if 'glove' in vec_file:
        return load_glove_txt(vec_file, verbose=verbose, dev=dev)
    else:
        print '\n\tError: unrecognized vectors. please add reader to embeddings.py\n'
        exit(3)


if __name__ == '__main__':
    # load vectors!
    #vectors_file = '/scratch/wboag/models/mimic-vectors.bin'
    #W = load_word2vec_binary(vectors_file, verbose=0, dev=False)
    #W = None

    # load vectors!
    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectors_file = os.path.join(homedir, 'models', 'glove.6B.100d.txt')
    #W = load_glove_txt(vectors_file, verbose=1, dev=False)
    #W = load_glove_txt(vectors_file, verbose=1, dev=True)
    #W = None

    print
    for w,v in sorted(W.items()):
        print w
    print
    print len(W)
    print len(W.values()[0])
    print

