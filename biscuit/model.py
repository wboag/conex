import sys
import os
from collections import defaultdict
from time import localtime, strftime
import random
import math
import io

import keras_ml
from embeddings import load_embeddings

import numpy as np
import nltk

from tools import flatten, save_list_structure, reconstruct_list
from tools import pickle_dump, load_pickled_obj
from tools import prose_partition
from tools import is_prose_sentence

from documents import labels as tag2id



# reverse this dict
id2tag = { v:k for k,v in tag2id.items() }




def print_features(f, label, feature_names):
    COLUMNS = 4
    feature_names = sorted(feature_names)
    print >>f, '\t    %s' % label
    start = 0
    for row in range(len(feature_names)/COLUMNS + 1):
        print >>f,'\t\t',
        for featname in feature_names[start:start+COLUMNS]:
            print >>f, '%-15s' % featname,
        print >>f, ''
        start += COLUMNS



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
            for label,vec in self._score['history'  ].items():
                print_vec(f, '%-16s'%label, vec)
            f.write(u'\n')
            for hyperparam,val in self._score['hyperparams'].items():
                f.write(unicode('\t%-15s: %s\n' % (hyperparam,val)))
            f.write(u'\n')
            f.write(unicode(self._score['model']))
            f.write(u'\n')
            f.write(unicode(self._score['train']['iob_conf']))
            print_vec(f, 'train iob precision', self._score['train']['iob_precision'])
            print_vec(f, 'train iob recall   ', self._score['train']['iob_recall'   ])
            print_vec(f, 'train iob f1       ', self._score['train']['iob_f1'       ])
            f.write(u'\n')
            f.write(unicode(self._score['train']['con_conf']))
            print_vec(f, 'train con precision', self._score['train']['con_precision'])
            print_vec(f, 'train con recall   ', self._score['train']['con_recall'   ])
            print_vec(f, 'train con f1       ', self._score['train']['con_f1'       ])
            if 'dev' in self._score:
                f.write(u'\n')
                f.write(unicode(self._score['dev'  ]['iob_conf']))
                print_vec(f, u'dev iob precision   ',self._score['dev']['iob_precision'])
                print_vec(f, u'dev iob recall      ',self._score['dev']['iob_recall'   ])
                print_vec(f, u'dev iob f1          ',self._score['dev']['iob_f1'       ])
                f.write(u'\n')
                f.write(unicode(self._score['dev'  ]['con_conf']))
                print_vec(f, u'dev con precision   ',self._score['dev']['con_precision'])
                print_vec(f, u'dev con recall      ',self._score['dev']['con_recall'   ])
                print_vec(f, u'dev con f1          ',self._score['dev']['con_f1'       ])
            f.write(u'\n')
            if self._training_files:
                f.write(u'\n')
                f.write(u'Training Files\n')
                print_files(f, self._training_files)
                f.write(u'\n')
            else:
                f.write(u'\n')
                f.write(u'Training Files: %d\n' % self._n_training_files)
                f.write(u'\n')
            f.write(u'-'*40)
            f.write(u'\n\n')

            # get output as full string
            contents = f.getvalue()
        return contents


    def serialize(self, filename, logfile=None):
        # Serialize the model
        pickle_dump(self, filename)

        # Describe training info?
        if logfile:
            #with open(logfile, 'a') as f:
            f = sys.stdout
            if logfile:
                self.log(logfile=logfile, model_file=filename)


    def __init__(self):
        # Classifiers
        self._is_trained = None
        self._clf        = None
        self._word_vocab = None
        self._char_vocab = None
        self._training_files = None
        self._n_training_files = None
        self._score      = {}


    def fit_from_documents(self, documents, hyperparams={}):
        """
        GalenModel::fit_from_documents()

        Train clinical concept extraction model using annotated data (files).

        @param notes.       A list of Document objects (containing text and annotations)
        @param hyperparams. A dict of CLI-specified hyperparameters
        @return             None
        """
        # Extract formatted data
        tokenized_sents  = flatten([d.getTokenizedSentences() for d in documents])
        labels           = flatten([d.getTokenLabels()        for d in documents])

        # Save training file info
        self._n_training_files = len(documents)
        if self._n_training_files < 100:
            self._training_files = [ d.getName() for d in documents ]

        # Call the internal method
        self.fit(tokenized_sents, labels, dev_split=0.10, hyperparams=hyperparams)


    def fit(self, tok_sents, tags, val_sents=None, val_tags=None, dev_split=None, hyperparams={}):
        '''
        GalenModel::fit()

        Train clinical concept extraction model using annotated data.

        @param tok_sents.   A list of sentences, where each sentence is tokenized
                              into words
        @param tags.        Parallel to `tokenized_sents`, 7-way labels for 
                              concept spans
        @param val_sents.   Validation data. Same format as tokenized_sents
        @param val_tags.    Validation data. Same format as iob_nested_labels
        @param dev_split.   A real number from 0 to 1
        @param hyperparams. A dict of CLI-specified hyperparameters
        '''
        # metadata
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # train classifier
        V_w, V_c, clf, dev_score = generic_train('all', tok_sents, tags,
                                                 val_sents=val_sents,val_labels=val_tags,
                                                 dev_split=dev_split,
                                                 hyperparams=hyperparams)
        self._is_trained = True
        self._word_vocab = V_w
        self._char_vocab = V_c
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
                  val_sents=None, val_labels=None, dev_split=None,
                  hyperparams={}):
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
    @param hyperparams.        A dict of CLI-specified hyperparameters
    '''

    global W

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

        #tokenized_sents   = train_sents
        #tags              = train_labels

    print '\tvectorizing words', p_or_n

    # build vocabulary of words & chars
    word_vocab = build_vocab(    train_sents    )
    char_vocab = build_vocab(sum(train_sents,[]))

    if os.path.exists(hyperparams['embeddings']):
        # load the word vectors based on input arguments
        W = load_embeddings(hyperparams['embeddings'])

        #W = { k:v[:4] for k,v in W.items() }
        dim = len(W.values()[0])
        W_init = np.random.rand(len(word_vocab),dim)
        for w,ind in word_vocab.items():
            if w in W:
                W_init[ind,:] = W[w]

        # how many words got initialized?
        w_vec = set(W.keys())
        w_voc = set(word_vocab.keys())
        both = len(w_voc&w_vec)
        tot  = len(w_voc)
        print '\t\tinit: %.3f (%d/%d)' % (float(both)/tot,both,tot)
    else:
        W_init = None
        print '\t\trandom initial embeddings'

    # L - number of lines
    # S - sentence length
    # W - word length

    # build matrix of words (LxS)
    train_X_word_ids = []
    for sent in train_sents:
        id_seq = [ (word_vocab[w] if w in word_vocab else word_vocab['oov'])
                   for w in sent                                             ]
        train_X_word_ids.append(id_seq)

    val_X_word_ids = []
    for sent in val_sents:
        id_seq = [ (word_vocab[w] if w in word_vocab else word_vocab['oov'])
                   for w in sent                                             ]
        val_X_word_ids.append(id_seq)

    # build tensor of characters (LxSxW)
    train_X_char_ids = []
    for sent in train_sents:
        seq_char_ids = []
        for word in sent:
            id_seq = [ (char_vocab[c] if c in char_vocab else char_vocab['oov'])
                       for c in word                                            ]
            assert id_seq != []
            seq_char_ids.append(id_seq)
        train_X_char_ids.append(seq_char_ids)

    val_X_char_ids = []
    for sent in val_sents:
        seq_char_ids = []
        for word in sent:
            id_seq = [ (char_vocab[c] if c in char_vocab else char_vocab['oov'])
                       for c in word                                            ]
            assert id_seq != []
            seq_char_ids.append(id_seq)
        val_X_char_ids.append(seq_char_ids)

    # vectorize IOB labels
    train_Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in train_labels ]
    val_Y_labels   = [ [tag2id[y] for y in y_seq] for y_seq in   val_labels ]

    print '\ttraining classifiers', p_or_n

    # Train classifier
    clf, dev_score  = keras_ml.train(train_X_word_ids = train_X_word_ids, 
                                     train_X_char_ids = train_X_char_ids, 
                                     train_Y_ids      = train_Y_labels, 
                                       val_X_word_ids = val_X_word_ids, 
                                       val_X_char_ids = val_X_char_ids,
                                       val_Y_ids      = val_Y_labels,
                                     tag2id           = tag2id, 
                                     W                = W_init,
                                     hyperparams      = hyperparams)

    return word_vocab, char_vocab, clf, dev_score



def generic_predict(p_or_n, tokenized_sents, word_vocab, char_vocab, clf):

    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []


    print '\tvectorizing words ' + p_or_n

    # build matrix of words (LxS)
    X_word_ids = []
    for sent in tokenized_sents:
        id_seq = [ (word_vocab[w] if w in word_vocab else word_vocab['oov'])
                   for w in sent                                             ]
        X_word_ids.append(id_seq)

    # build tensor of characters (LxSxW)
    X_char_ids = []
    for sent in tokenized_sents:
        seq_char_ids = []
        for word in sent:
            id_seq = [ (char_vocab[c] if c in char_vocab else char_vocab['oov'])
                       for c in word                                            ]
            assert id_seq != []
            seq_char_ids.append(id_seq)
        X_char_ids.append(seq_char_ids)

    print '\tpredicting  labels ' + p_or_n

    # Predict labels
    predictions = keras_ml.predict(clf, X_word_ids, X_char_ids)

    # Format labels from output
    return predictions



def build_vocab(list_of_seqs):
    vocab = {}
    for seq in list_of_seqs:
        for unit in seq:
            if unit not in vocab:
                vocab[unit] = len(vocab) + 1
    vocab['oov'] = len(vocab)
    return vocab



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
    for row in range(int(math.ceil(float(len(vec))/COLUMNS))):
        for featname in vec[start:start+COLUMNS]:
            f.write(unicode('%7.3f' % featname))
        f.write(u'\n')
        start += COLUMNS
