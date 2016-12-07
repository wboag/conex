######################################################################
#  CliNER - model.py                                                 #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: Define the model for clinical concept extraction.        #
######################################################################

__author__ = 'Willie Boag'
__date__   = 'Aug. 15, 2016'


from collections import defaultdict
import os
import sys
import io
import random
from time import localtime, strftime

import keras_ml
from documents import labels as tag2id, id2tag
from tools import flatten
from tools import pickle_dump, load_pickled_obj



class GalenModel:

    def log(self, logfile, model_file=None):
        '''
        GalenMdoel::log()

        Log training information of model.

        @param logfile.     A file to append the log information to.
        @param model_file.  A path to optionally identify where the model was saved.

        @return None
        '''
        if not self._log:
            log = self.__log_str(model_file)
        else:
            log = self._log

        with open(logfile, 'a') as f:
            print >>f, log


    def __log_str(self, model_file=None):
        '''
        GalenModel::__log_str()

        Build a string of information about training for the model's log file.

        @param model_file.  A path to optionally identify where the model was saved.

        @return  A string of the model's training information
        '''
        assert self._is_trained, 'GalenModel not trained'
        with io.StringIO() as f:
            f.write(u'\n')
            f.write(unicode('-'*40))
            f.write(u'\n\n')
            if model_file:
                f.write(unicode('model         : %s\n' % os.path.abspath(model_file)))
                f.write(u'\n')
            f.write(unicode('training began: %s\n' % self._time_train_begin))
            f.write(unicode('training ended: %s\n' % self._time_train_end))
            f.write(u'\n')
            f.write(u'scores\n')
            print_vec(f, 'train precision', self._score['train']['precision'])
            print_vec(f, 'train recall   ', self._score['train']['recall'   ])
            print_vec(f, 'train f1       ', self._score['train']['f1'       ])
            if 'dev' in self._score:
                print_vec(f, u'dev precision   ', self._score['dev']['precision'])
                print_vec(f, u'dev recall      ', self._score['dev']['recall'   ])
                print_vec(f, u'dev f1          ', self._score['dev']['f1'       ])
            for label,vec in self._score['history'  ].items():
                print_vec(f, '%-16s'%label, vec)
            f.write(u'\n')
            if self._training_files:
                f.write(u'\n')
                f.write(u'Training Files\n')
                print_files(f, self._training_files)
                f.write(u'\n')
            f.write(u'-'*40)
            f.write(u'\n\n')

            # get output as full string
            contents = f.getvalue()
        return contents


    def __init__(self):
        """
        GalenModel::__init__()

        Instantiate a GalenModel object.
        """
        self._is_trained     = None
        self._clf            = None
        self._vocab          = None
        self._training_files = None
        self._log            = None


    def fit_from_documents(self, documents):
        """
        GalenModel::fit_from_documents()

        Train clinical concept extraction model using annotated data (files).

        @param notes. A list of Document objects (containing text and annotations)
        @return       None
        """
        # Extract formatted data
        tokenized_sents  = flatten([d.getTokenizedSentences() for d in documents])
        labels           = flatten([d.getTokenLabels()        for d in documents])

        # Call the internal method
        self.fit(tokenized_sents, labels, dev_split=0.10)

        self._training_files = [ d.getName() for d in documents ]


    def fit(self, tok_sents, tags, val_sents=None, val_tags=None, dev_split=None):
        '''
        GalenModel::fit()

        Train clinical concept extraction model using annotated data.

        @param tok_sents.  A list of sentences, where each sentence is tokenized
                             into words
        @param tags.       Parallel to `tokenized_sents`, 7-way labels for 
                             concept spans
        @param val_sents.  Validation data. Same format as tokenized_sents
        @param val_tags.   Validation data. Same format as iob_nested_labels
        @param dev_split.  A real number from 0 to 1
        '''
        # metadata
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # train classifier
        voc, clf, dev_score = generic_train('all', tok_sents, tags,
                                            val_sents=val_sents, val_labels=val_tags,
                                            dev_split=dev_split)
        self._is_trained = True
        self._vocab = voc
        self._clf   = clf
        self._score = dev_score

        # metadata
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())

        self._log = self.__log_str()


    def predict_classes_from_document(self, document):
        """
        GalenModel::predict_classes_from_documents()

        Predict concept annotations for a given document

        @param note. A Document object (containing text and annotations)
        @return      List of predictions
        """
        # Extract formatted data
        tokenized_sents  = note.getTokenizedSentences()

        return self.predict_classes(tokenized_sents)


    def predict_classes(self, tokenized_sents):
        """
        GalenModel::predict_classes()

        Predict concept annotations for unlabeled, tokenized sentences

        @param tokenized_sents. A list of sentences, where each sentence is tokenized
                                  into words
        @return                  List of predictions
        """
        # Predict labels for prose
        num_pred = generic_predict('all'                   ,
                                   tokenized_sents         ,
                                   vocab    = self._vocab  ,
                                   clf      = self._clf    )
        iob_pred = [ [id2tag[p] for p in seq] for seq in num_pred ]

        return iob_pred



def generic_train(p_or_n, tokenized_sents, tags,
                  val_sents=None, val_labels=None, dev_split=None):
    '''
    generic_train()

    Train a model that works for both prose and nonprose

    @param p_or_n.             A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents.    A list of sentences, where each sentence is tokenized
                                 into words
    @param iob_nested_labels.  Parallel to `tokenized_sents`, 7-way labels for 
                                 concept spans
    @param val_sents.          Validation data. Same format as tokenized_sents
    @param val_labels.         Validation data. Same format as iob_nested_labels
    @param dev_split.          A real number from 0 to 1
    '''
    # Must have data to train on
    if len(tokenized_sents) == 0:
        raise Exception('Training must have %s training examples' % p_or_n)

    # if you should split the data into train/dev yourself
    #if (not val_sents) and (dev_split > 0.0) and (len(tokenized_sents)>1000):
    if (not val_sents) and (dev_split > 0.0) and (len(tokenized_sents)>10):

        p = int(dev_split*100)
        print '\tCreating %d/%d train/dev split' % (100-p,p)

        perm = range(len(tokenized_sents))
        random.shuffle(perm)
        tokenized_sents = [ tokenized_sents[i] for i in perm ]
        tags            = [            tags[i] for i in perm ]

        ind = int(dev_split*len(tokenized_sents))

        val_sents   = tokenized_sents[:ind ]
        train_sents = tokenized_sents[ ind:]

        val_labels   = tags[:ind ]
        train_labels = tags[ ind:]

        tokenized_sents   = train_sents
        tags              = train_labels


    print '\tvectorizing words', p_or_n

    #tokenized_sents   = train_sents[ :2]
    #iob_nested_labels = train_labels[:2]

    # count word frequencies to determine OOV
    freq = defaultdict(int)
    for sent in tokenized_sents:
        for w in sent:
            freq[w] += 1

    # determine OOV based on % of vocab or minimum word freq threshold
    if len(freq) < 100:
        lo = len(freq)/20
        oov = set([ w for w,f in sorted(freq.items(), key=lambda t:t[1]) ][:lo])
    else:
        lo = 2
        oov = set([ w for w,f in freq.items() if (f <= lo) ])

    '''
    for w in oov:
        print w
    print 
    print len(oov)
    print len(freq)
    '''
    '''
    val = None
    for w,f in sorted(freq.items(), key=lambda t:t[1]):
        if val != f:
            val = f
            print
        print '%8d  %s' % (f,w)
    exit()
    '''

    # build vocabulary of words
    vocab = {}
    for sent in tokenized_sents:
        for w in sent:
            if (w not in vocab) and (w not in oov):
                vocab[w] = len(vocab) + 1
    vocab['oov'] = len(vocab) + 1

    # vectorize tokenized sentences
    X_seq_ids = []
    for sent in tokenized_sents:
        id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
        X_seq_ids.append(id_seq)

    # vectorize IOB labels
    Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in tags ]

    print '\ttraining classifiers', p_or_n

    #val_sents  = val_sents[ :5]
    #val_labels = val_labels[:5]

    # if there is specified validation data, then vectorize it
    if val_sents:
        # vectorize validation X
        val_X = []
        for sent in val_sents:
            id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
            val_X.append(id_seq)
        # vectorize validation Y
        val_Y = [ [tag2id[y] for y in y_seq] for y_seq in val_labels ]
        # rename 
        val_sents  = val_X
        val_labels = val_Y

    # train using lstm
    clf, dev_score  = keras_ml.train(X_seq_ids, Y_labels, tag2id, 
                                     val_X_ids=val_sents, val_Y_ids=val_labels)

    return vocab, clf, dev_score



def generic_predict(p_or_n, tokenized_sents, vocab, clf):
    '''
    generic_predict()

    Train a model that works for both prose and nonprose

    @param p_or_n.          A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents. A list of sentences, where each sentence is tokenized
                              into words
    @param vocab.           A dictionary mapping word tokens to numeric indices.
    @param clf.             An encoding of the trained keras model.
    '''

    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []

    print '\tvectorizing words ' + p_or_n

    # vectorize tokenized sentences
    X_seq_ids = []
    for sent in tokenized_sents:
        id_seq = []
        for w in sent:
            if w in vocab:
                id_seq.append(vocab[w])
            else:
                id_seq.append(vocab['oov'])
        X_seq_ids.append(id_seq)

    print '\tpredicting  labels ' + p_or_n

    # Predict labels
    predictions = keras_ml.predict(clf, X_seq_ids)

    # Format labels from output
    return predictions



def print_files(f, file_names):
    '''
    print_files()

    Pretty formatting for listing the training files in a log.

    @param f.           An open file stream to write to.
    @param file_names.  A list of filename strings.
    '''
    COLUMNS = 4
    file_names = sorted(file_names)
    start = 0
    for row in range(len(file_names)/COLUMNS + 1):
        f.write(u'\t\t')
        for featname in file_names[start:start+COLUMNS]:
            f.write(unicode('%-15s' % featname))
        f.write(u'\n')
        start += COLUMNS



def print_vec(f, label, vec):
    '''
    print_vec()

    Pretty formatting for displaying a vector of numbers in a log.

    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param vec.    A numpy array of the numbers to display.
    '''
    COLUMNS = 7
    start = 0
    f.write(unicode('\t%-10s: ' % label))
    if type(vec) != type([]):
        vec = vec.tolist()
    for row in range(len(vec)/COLUMNS):
        for featname in vec[start:start+COLUMNS]:
            f.write(unicode('%7.3f' % featname))
        f.write(u'\n')
        start += COLUMNS

