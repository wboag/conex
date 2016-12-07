######################################################################
#  CliCon - crf.py                                                   #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Implement CRF (using python-crfsuite)                    #
######################################################################


import sys
import os
import tempfile
import pycrfsuite

count = 0

tmp_dir = '/tmp'

def format_features(rows, labels=None):

    retVal = []

    # For each line
    for i,line in enumerate(rows):

        # For each word in the line
        for j,features in enumerate(line):

            # Nonzero dimensions
            inds  = features.nonzero()[1]

            # If label exists
            values = []
            if labels:
                values.append( str(labels[i][j]) )

            # Value for each dimension
            for k in inds:
                values.append( '%d=%d' %  (k, features[0,k]) )

            retVal.append("\t".join(values).strip())

        # Sentence boundary seperator
        retVal.append('')


    '''
    # Sanity check
    global count
    if labels:
        out_f = 'a.txt' + str(count)
        start =  0 # 2
    else:
        out_f = 'b.txt' + str(count)
        start = 0
    count += 1
    with open(out_f, 'w') as f:
        for line in retVal:
            print >>f, line[start:]
    '''

    return retVal




def pycrf_instances(fi, labeled):
    xseq = []
    yseq = []

    # Skip first element
    if labeled:
        begin = 1
    else:
        begin = 0

    for line in fi:

        line = line.strip('\n')
        if not line:
            # An empty line presents an end of a sequence.
            if labeled:
                yield xseq, tuple(yseq)
            else:
                yield xseq

            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item to the item sequence.
        feats = fields[begin:]
        xseq.append(feats)

        # Append the label to the label sequence.
        if labeled:
            yseq.append(fields[0])



def train(X, Y):

    # Sanity Check detection: features & label
    #with open('a','w') as f:
    #    for xline,yline in zip(X,Y):
    #        for x,y in zip(xline,yline):
    #            print >>f, y, '\t', x.nonzero()[1][0]
    #        print >>f

    # Format features fot crfsuite
    feats = format_features(X,Y)

    # Create a Trainer object.
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in pycrf_instances(feats, labeled=True):
        trainer.append(xseq, yseq)

    # Train the model
    os_handle,tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp")
    trainer.train(tmp_file)

    # Read the trained model into a string
    model = ''
    with open(tmp_file, 'r') as f:
        model = f.read()
    os.close(os_handle)

    # Remove the temporary file
    os.remove(tmp_file)

    return model




def predict(clf, X):

    # Format features fot crfsuite
    feats = format_features(X)

    # Dump the model into a temp file
    os_handle,tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp")
    with open(tmp_file, 'wb') as f:
        f.write(clf)

    # Create the Tagger object
    tagger = pycrfsuite.Tagger()
    tagger.open(tmp_file)

    # Remove the temp file
    os.close(os_handle)
    os.remove(tmp_file)

    # Tag the sequence
    retVal = []
    Y = []
    i = 0
    for xseq in pycrf_instances(feats, labeled=False):
        yseq = [ int(n) for n in tagger.tag(xseq) ]
        #print 
        #print i
        #print len(X[i])
        #print len(xseq)
        assert len(X[i]) == len(xseq)
        i += 1
        retVal += list(yseq)
        Y.append(list(yseq))

    # Sanity Check detection: feature & label predictions
    #with open('a','w') as f:
    #    for x,y in zip(xseq,Y):
    #        x = x[0]
    #        print >>f, y, '\t', x[:-2]
    #    print >>f

    #return retVal
    return Y


