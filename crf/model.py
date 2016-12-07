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
import numpy as np

from sklearn.feature_extraction import DictVectorizer

import crf
from documents import labels as tag2id, id2tag
from tools import flatten, save_list_structure, reconstruct_list
from tools import pickle_dump, load_pickled_obj



class CRF:

    def log(self, logfile, model_file=None):
        '''
        CRF::log()

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
        CRF::__log_str()

        Build a string of information about training for the model's log file.

        @param model_file.  A path to optionally identify where the model was saved.

        @return  A string of the model's training information
        '''
        assert self._is_trained, 'CRF not trained'
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
            tags = [ t[0] for t in sorted(self._tag2ind.items(), key=lambda t:t[1])]
            str_tags = ' '.join(map(str,tags))
            f.write(u'tags: %s' % str_tags)
            f.write(u'\n\n')
            f.write(u'scores\n')
            print_vec(f, 'train precision', self._score['train']['precision'])
            print_vec(f, 'train recall   ', self._score['train']['recall'   ])
            print_vec(f, 'train f1       ', self._score['train']['f1'       ])
            f.write(unicode(self._score['train']['confusion']))
            f.write(u'\n')
            if 'dev' in self._score:
                print_vec(f, u'dev precision   ', self._score['dev']['precision'])
                print_vec(f, u'dev recall      ', self._score['dev']['recall'   ])
                print_vec(f, u'dev f1          ', self._score['dev']['f1'       ])
                f.write(unicode(self._score['dev']['confusion']))
            if 'history' in self._score:
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
        CRF::__init__()

        Instantiate a CRF object.
        """
        self._is_trained     = None
        self._clf            = None
        self._vec            = None
        self._training_files = None
        self._log            = None


    def fit_from_documents(self, documents):
        """
        CRF::fit_from_documents()

        Train clinical concept extraction model using annotated data (files).

        @param notes. A list of Document objects (containing text and annotations)
        @param val_sents.  Validation data. Same format as tokenized_sents
        @param val_tags.   Validation data. Same format as iob_nested_labels
        @param dev_split.  A real number from 0 to 1
        @return       None
        """
        # Extract formatted data
        tokenized_sents  = flatten([d.getTokenizedSentences() for d in documents])
        labels           = flatten([d.getTokenLabels()        for d in documents])

        self._training_files = [ d.getName() for d in documents ]

        # Call the internal method
        self.fit(tokenized_sents, labels, dev_split=0.10)



    def fit(self, tok_sents, tags, val_sents=None, val_tags=None, dev_split=None):
        '''
        CRF::fit()

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
        vec, clf, tag2ind, dev_score = generic_train('all', tok_sents, tags,
                                                     val_sents=val_sents, val_labels=val_tags,
                                                     dev_split=dev_split)
        self._is_trained = True
        self._vec     = vec
        self._clf     = clf
        self._tag2ind = tag2ind
        self._score   = dev_score

        # metadata
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())

        self._log = self.__log_str()


    def predict_classes_from_document(self, document):
        """
        CRF::predict_classes_from_documents()

        Predict concept annotations for a given document

        @param note. A Document object (containing text and annotations)
        @return      List of predictions
        """
        # Extract formatted data
        tokenized_sents  = document.getTokenizedSentences()

        return self.predict_classes(tokenized_sents)


    def predict_classes(self, tokenized_sents):
        """
        CRF::predict_classes()

        Predict concept annotations for unlabeled, tokenized sentences

        @param tokenized_sents. A list of sentences, where each sentence is tokenized
                                  into words
        @return                  List of predictions
        """
        # Predict labels for prose
        iob_pred = generic_predict('all'                    ,
                                   tokenized_sents          ,
                                   vec      = self._vec     ,
                                   tag2ind  = self._tag2ind ,
                                   crf_model= self._clf     )
        return iob_pred



def generic_train(p_or_n, tokenized_sents, tags,
                  val_sents=None, val_labels=None, dev_split=None):
    '''
    generic_train()

    Train a model that works for both prose and nonprose

    @param p_or_n.             A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents.    A list of sentences, where each sentence is tokenized
                                 into words
    @param tags.               Parallel to `tokenized_sents`, 7-way labels for 
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

        #'''
        perm = range(len(tokenized_sents))
        random.shuffle(perm)
        tokenized_sents = [ tokenized_sents[i] for i in perm ]
        tags            = [            tags[i] for i in perm ]
        #'''

        ind = int(dev_split*len(tokenized_sents))

        val_sents   = tokenized_sents[:ind ]
        train_sents = tokenized_sents[ ind:]

        val_labels   = tags[:ind ]
        train_labels = tags[ ind:]

        tokenized_sents   = train_sents
        tags              = train_labels

    ######################################################################
    #                            FEATURE ENGINEERING                     #
    ######################################################################

    text_features = extract_features(tokenized_sents)

    ######################################################################
    #                         FORMATTING DATA                            #
    ######################################################################

    # text features -> sparse numeric design matrix
    dvec = DictVectorizer()

    # convert text features to design matrix
    offsets = save_list_structure(text_features)
    flat_text_features = flatten(text_features)
    flat_train_X = dvec.fit_transform(flat_text_features)

    # vectorize labels
    flat_tags = flatten(tags)

    tag2ind = { tag:i for i,tag in enumerate(set(flat_tags)) }
    ind2tag = { i:tag for tag,i in tag2ind.items()           }
   
    flat_train_Y = [ tag2ind[tag] for tag in flat_tags ]
   
    # reconstruct list structures
    train_X = reconstruct_list( list(flat_train_X) , offsets)
    train_Y = reconstruct_list(      flat_train_Y  , offsets)

    # build CRF model
    crf_model = crf.train(train_X, train_Y)

    # how well does the model fit the training data?
    score = {}
    score['train'] = compute_stats('train', crf_model, train_X, train_Y, len(tag2ind))

    if val_sents:
        val_text_features = extract_features(val_sents)

        val_offsets = save_list_structure(val_text_features)
        val_flat_text_features = flatten(val_text_features)
        flat_val_X = dvec.transform(val_flat_text_features)

        val_X = reconstruct_list( list(flat_train_X) , offsets)

        flat_val_tags = flatten(val_labels)
        flat_val_Y = [ tag2ind[tag] for tag in flat_val_tags ]
        val_Y = reconstruct_list(flat_val_Y, val_offsets)

        score['dev'] = compute_stats('dev', crf_model, val_X, val_Y, len(tag2ind))

    return dvec, crf_model, tag2ind, score



def generic_predict(p_or_n, sents, vec, tag2ind, crf_model):
    '''
    generic_predict()

    Train a model that works for both prose and nonprose

    @param p_or_n.          A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents. A list of sentences, where each sentence is tokenized
                              into words
    @param vec.             A dictionary mapping word tokens to numeric indices.
    @param clf.             An encoding of the trained keras model.
    '''

    # If nothing to predict, skip actual prediction
    if len(sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []

    ######################################################################
    #                        FEATURE ENGINEERING                         #
    ######################################################################

    text_features = extract_features(sents)

    ######################################################################
    #                         FORMATTING DATA                            #
    ######################################################################

    # convert text features to design matrix
    flat_pred_X = vec.transform(flatten(text_features))

    # reconstruct list structures
    offsets = save_list_structure(text_features)
    pred_X = reconstruct_list( list(flat_pred_X) , offsets)

    ######################################################################
    #                            PREDICTING                              #
    ######################################################################

    # make the predictions
    ind2tag = { i:tag for tag,i in tag2ind.items() }
    pred_Y = crf.predict(crf_model, pred_X)
    pred_tags = [ [ind2tag[p] for p in P] for P in pred_Y ]

    assert len(pred_Y) == len(pred_X)
    for i in range(len(pred_Y)):
        assert len(pred_Y[i]) == len(pred_X[i])

    # correct illegal predictions (AKA bad I -> legal B)
    for lineno,preds in enumerate(pred_tags):
        for i in range(len(preds)):
            if preds[i][0] == 'I':
                if preds[i-1][0] == 'O' or preds[i][1:]!=preds[i][1:]:
                    preds[i] = 'B-%s' % preds[i][2:]

    # assert proper formatting
    for lineno,preds in enumerate(pred_tags):
        for i in range(len(preds)):
            if preds[i][0] == 'I':
                assert preds[i-1][0] != 'O' and preds[i][1:]==preds[i][1:]

    # Format labels from output
    return pred_tags



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




def extract_features(sents):

    prev_N = 3
    copies = 10

    # get some features for each word of each sentence
    text_features = []
    for lineno,sent in enumerate(sents):
        features_list = []

        for i,w in enumerate(sent):
            features = {'dummy':1}

            # previous words (especially helpful for beginning-of-sentence words
            for j in range(1,prev_N+1):
                if i-j >= 0:
                    prev_word = sent[i-j]
                else:
                    prev_word = '<PAD>'
                for k in range(copies):
                    features[('prev-unigram-%d'%j,k,prev_word)] = 1

            # unigram (note: crfsuite has weird issues when given too few feats)
            for j in range(copies):
                features[('unigram',j,w.lower())] = 1

            # next words (especially helpful for end-of-sentence words
            for j in range(1,prev_N+1):
                if i+j < len(sent):
                    next_word = sent[i+j]
                else:
                    next_word = '<PAD>'
                for k in range(copies):
                    features[('next-unigram-%d'%j,k,next_word)] = 1

            '''
            print 'w: ', w
            print features
            print
            '''

            features_list.append(features)

        text_features.append(features_list)
        #print

    return text_features



def compute_stats(label, model, X, Y, num_tags):
    '''
    compute_stats()

    Compute the P, R, and F for a given model on some data.

    @param label.        A name for the data (e.g. "train" or "dev")
    @param model.        A trained model.
    @param X.            A formatted collection of input examples
    @param Y.            A list of list of tags - the labels to X.
    '''
    # predict label probabilities
    predictions = crf.predict(model, X)

    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(predictions,Y):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

    # print confusion matrix
    with io.StringIO() as f:
        f.write(u'\n\n')
        f.write(unicode(label))
        f.write(u'\n')
        f.write(u'      ')
        for i in range(num_tags):
            f.write(u'%4d ' % i)
        f.write(u' (gold)\n')
        for i in range(num_tags):
            f.write(u'%2d' % i +u'    ')
            for j in range(num_tags):
                f.write(u'%4d ' % confusion[i][j])
            f.write(u'\n')
        f.write(u'(pred)\n')
        f.write(u'\n\n')
        confusion_str = f.getvalue()
    print confusion_str

    precision = np.zeros(num_tags)
    recall    = np.zeros(num_tags)
    f1        = np.zeros(num_tags)

    for i in range(num_tags):
        correct    =     confusion[i,i]
        num_pred   = sum(confusion[i,:])
        num_actual = sum(confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        precision[i] = p
        recall[i]    = r
        f1[i]        = (2*p*r) / (p + r + 1e-9)

    scores = {}
    scores['precision'] = precision
    scores['recall'   ] = recall
    scores['f1'       ] = f1
    scores['confusion'] = confusion_str

    return scores


