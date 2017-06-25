
import numpy as np
import os
import time
import random
import sys

import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.layers import Lambda, Masking
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping

from tools import matrix_max

hierarchical_lstm = None


def train(train_X_word_ids   , train_X_char_ids   , train_Y_ids   , tag2id,
            val_X_word_ids=[],   val_X_char_ids=[],   val_Y_ids=[], W=None, epochs=100):

    word_input_dim =         matrix_max(train_X_word_ids)  + 1
    char_input_dim = max(map(matrix_max,train_X_char_ids)) + 1

    word_maxlen = 32
    char_maxlen = 16

    num_tags   = len(tag2id)
    nb_train_samples = len(train_X_word_ids)
    nb_val_samples   = len(  val_X_word_ids)

    hierarchical_lstm = create_model(word_input_dim, char_input_dim,
                                     word_maxlen   , char_maxlen   ,
                                     num_tags      , W             )

    # turn each id in Y_ids into a onehot vector
    train_Y_seq_onehots = [to_categorical(y, num_classes=num_tags) for y in train_Y_ids]
    val_Y_seq_onehots   = [to_categorical(y, num_classes=num_tags) for y in   val_Y_ids]

    # TODO - consider batching here if all data is too big for ram in dense matrix form

    # format X and Y data
    train_X_char = build_X_char_matrix(train_X_char_ids, nb_train_samples, 
                                       word_maxlen, char_maxlen)
    train_X_word = build_X_word_matrix(train_X_word_ids, nb_train_samples, word_maxlen)

    train_Y = create_data_matrix_Y(train_Y_seq_onehots, nb_train_samples, 
                                  word_maxlen, num_tags)

    val_X_char = build_X_char_matrix(val_X_char_ids, nb_val_samples, 
                                       word_maxlen, char_maxlen)
    val_X_word = build_X_word_matrix(val_X_word_ids, nb_val_samples, word_maxlen)

    val_Y = create_data_matrix_Y(val_Y_seq_onehots, nb_val_samples, 
                                  word_maxlen, num_tags)

    print 
    print 'V_c:  ', char_input_dim
    print 'V_w:  ', word_input_dim
    print 'char: ', train_X_char.shape
    print 'word: ', train_X_word.shape
    print 

    #S = hierarchical_lstm.summary()
    #print S

    print 
    print 'training begin'
    print 

    batch_size = 512

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # merge the train/val back together to feed in (allows user to specify each, though)
    train_size = train_X_word.shape[0]
    val_size   =   val_X_word.shape[0]
    val_frac = val_size/float(train_size+val_size)
    X_word = np.concatenate((train_X_word, val_X_word), axis=0)
    X_char = np.concatenate((train_X_char, val_X_char), axis=0)
    Y      = np.concatenate((train_Y     , val_Y     ), axis=0)

    # fit the model
    history = hierarchical_lstm.fit({'char': X_char, 'word':X_word},
                                    Y                                  ,
                                    batch_size       = batch_size      ,
                                    epochs           = epochs          ,
                                    validation_split = val_frac        ,
                                    callbacks        = [early_stopping],
                                    verbose          = 1               )

    print 'training done'

    ######################################################################

    # how many words in the vocabulary?
    W_shape = W.shape

    hyperparams = (word_input_dim, char_input_dim, num_tags,
                   word_maxlen   , char_maxlen   , W_shape )

    # information about fitting the model
    scores = {}
    scores['history'] = history.history

    scores['train'] = compute_stats('train', hierarchical_lstm, hyperparams,
                                    train_X_word, train_X_char, train_Y_ids)
    if val_X_word_ids:
        scores['dev'] = compute_stats('dev', hierarchical_lstm, hyperparams,
                                      val_X_word, val_X_char, val_Y_ids)

    ######################################################################

    # needs to return something pickle-able (so get binary serialized string)
    tmp_file = 'tmp_keras_weights-%d' % random.randint(0,10000)
    hierarchical_lstm.save_weights(tmp_file)
    with open(tmp_file, 'rb') as f:
        hierarchical_lstm_str = f.read()
    os.remove(tmp_file)

    '''
    # Verify encoded correctly
    lstm = create_model(word_input_dim, char_input_dim,
                        word_maxlen   , char_maxlen   ,
                        num_tags      , W_shape=W_shape)

    p = hierarchical_lstm.predict({'char': X_char, 'word':X_word},
                                  batch_size=batch_size)
    exit()
    '''

    # return model back to cliner
    keras_model_tuple = (hierarchical_lstm_str,
                         word_input_dim, char_input_dim,
                         num_tags,
                         word_maxlen, char_maxlen,
                         W_shape)
    return keras_model_tuple, scores





def predict(keras_model_tuple, X_word_ids, X_char_ids):

    global hierarchical_lstm

    # unpack model metadata
    #lstm_model_str, input_dim, num_tags, maxlen = keras_model_tuple
    hierarchical_lstm_str = keras_model_tuple[0]
    word_input_dim        = keras_model_tuple[1]
    char_input_dim        = keras_model_tuple[2]
    num_tags              = keras_model_tuple[3]
    word_maxlen           = keras_model_tuple[4]
    char_maxlen           = keras_model_tuple[5]
    W_shape               = keras_model_tuple[6]

    if not hierarchical_lstm:
        print '\t\tloading model from disk'

        # build LSTM
        hierarchical_lstm = create_model(word_input_dim, char_input_dim ,
                                         word_maxlen   , char_maxlen    ,
                                         num_tags      , W_shape=W_shape)

        #'''
        # load weights from serialized file
        tmp_file = 'tmp_keras_weights_pred-%d' % random.randint(0,100000)
        with open(tmp_file, 'wb') as f:
            f.write(hierarchical_lstm_str)
        hierarchical_lstm.load_weights(tmp_file)
        os.remove(tmp_file)
        #'''

    # format X and Y data
    nb_samples = len(X_word_ids)
    X_char = build_X_char_matrix(X_char_ids, nb_samples, word_maxlen, char_maxlen)
    X_word = build_X_word_matrix(X_word_ids, nb_samples, word_maxlen)

    #print X_word.shape
    #print X_char.shape
    #exit()

    # Predict tags using LSTM
    batch_size = 128
    p = hierarchical_lstm.predict({'char': X_char, 'word':X_word},
                                  batch_size=batch_size)

    p = p

    # decode (should really do a viterbi)
    predictions = []
    for i in range(nb_samples):
        num_words = len(X_word_ids[i])
        if num_words <= word_maxlen:
            tags = p[i,word_maxlen-num_words:].argmax(axis=1)
            predictions.append(tags.tolist())
        else:
            # if the sentence had more words than the longest sentence
            #   in the training set
            residual_zeros = [ 0 for _ in range(num_words-word_maxlen) ]
            padded = list(p[i].argmax(axis=1)) + residual_zeros
            predictions.append(padded)

    return predictions




def create_model(word_input_dim, char_input_dim,
                 word_maxlen   , char_maxlen   ,
                 num_tags                      , 
                 W=None        , W_shape=None  ):

    # hyperparams
    char_emb_size  = 25
    c_seq_emb_size = 25
    wlstm1_size    = 100
    wlstm2_size    = 100

    # pretrained word embeddings
    if W is not None:
        word_emb_size  = W.shape[1]
        W_init = [W]
    elif W_shape is not None:
        word_emb_size  = W_shape[1]
        W_init = [np.random.rand(W_shape[0], W_shape[1])]
    else:
        word_emb_size  = 100
        #W = np.random.rand(word_input_dim,word_emb_size)
        W_init = None

    num_tags_inner = num_tags

    # character-level LSTM encoder
    char_input = Input(shape=(char_maxlen,), dtype='int32')
    char_embedding = Embedding(output_dim=char_emb_size,
                               input_dim=char_input_dim,
                               input_length=char_maxlen,
                               mask_zero=True)(char_input)
    char_lstm_f    = LSTM(units=c_seq_emb_size                  )(char_embedding)
    char_lstm_r    = LSTM(units=c_seq_emb_size,go_backwards=True)(char_embedding)
    char_lstm_fr = Concatenate()([char_lstm_f,char_lstm_r])
    char_encoder_fr = Model(inputs=char_input, outputs=char_lstm_fr)

    # apply char-level encoder to every char sequence (word)
    char_seqs = Input(shape=(word_maxlen,char_maxlen),dtype='int32',name='char')
    encoded_char_fr_states = TimeDistributed(char_encoder_fr)(char_seqs)
    m_encoded_char_fr_states = Masking(0.0)(encoded_char_fr_states)

    # apply embeddings layer to every word
    word_seqs = Input(shape=(word_maxlen,), dtype='int32', name='word')
    word_embedding = Embedding(output_dim=word_emb_size,
                               input_dim=word_input_dim,
                               input_length=word_maxlen,
                               weights=W_init,
                               mask_zero=True)(word_seqs)

    # combine char-level encoded states WITH word embeddings
    word_feats = Concatenate(axis=-1)([m_encoded_char_fr_states, word_embedding])
    #word_feats = encoded_char_fr_states
    #word_feats = word_embedding

    # Dropout
    word_feats_d = TimeDistributed(Dropout(0.5))(word_feats)

    # word-level LSTM
    word_lstm_f1 = LSTM(units=wlstm1_size, return_sequences=True
                                        )(word_feats_d)
    word_lstm_r1 = LSTM(units=wlstm1_size, return_sequences=True,
                       go_backwards=True)(word_feats_d)
    word_lstm_fr1 = Concatenate(axis=-1)([word_lstm_f1, word_lstm_r1])

    # Predict labels using the sequence of word encodings
    orig_pred = TimeDistributed( Dense(units=num_tags_inner,
                                       activation='softmax' )  )(word_lstm_fr1)

    # TODO: crf layer
    pass

    model = Model( inputs=[char_seqs,word_seqs],
                   outputs=[orig_pred]  )

    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model



def compute_stats(label, lstm_model, hyperparams, X_word, X_char, Y_ids):
    '''
    compute_stats()
    Compute the P, R, and F for a given model on some data.
    @param label.        A name for the data (e.g. "train" or "dev")
    @param lstm_model.   The trained Keras model
    @param hyperparams.  A tuple of values for things like num_tags and batch_size
    @param X_word        A formatted collection of word inputs
    @param X_char        A formatted collection of character inputs
    @param Y_ids.        A list of list of tags - the labels to X.
    '''
    # un-pack hyperparameters
    word_input_dim,char_input_dim,num_tags,word_maxlen,char_maxlen,W_shape = hyperparams

    # predict label probabilities
    batch_size = 512
    pred = lstm_model.predict({'char':X_char,'word':X_word}, batch_size=batch_size)

    # choose the highest-probability labels
    nb_samples = len(Y_ids)
    predictions = []
    for i in range(nb_samples):
        num_words = len(Y_ids[i])
        tags = pred[i,word_maxlen-num_words:].argmax(axis=1)
        predictions.append(tags.tolist())

    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(predictions,Y_ids):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

    # print confusion matrix
    print '\n'
    print label
    print ' '*6,
    for i in range(num_tags):
        print '%4d' % i,
    print ' (gold)'
    for i in range(num_tags):
        print '%2d' % i, '   ',
        for j in range(num_tags):
            print '%4d' % confusion[i][j],
        print
    print '(pred)'
    print '\n'

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

    return scores






def build_X_word_matrix(X_ids, nb_samples, maxlen):
    X = np.zeros((nb_samples, maxlen))
    for i in range(nb_samples):
        cur_len = len(X_ids[i])
        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen
        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        X[i, maxlen - cur_len:] = X_ids[i][:maxlen]
    return X



def build_X_char_matrix(X_char_ids, nb_samples, word_maxlen, char_maxlen):
    X = np.zeros((nb_samples, word_maxlen, char_maxlen))

    for i in range(nb_samples):
        word_len = len(X_char_ids[i])

        # in prediction, could see longer sentence
        if word_maxlen-word_len < 0:
            word_len = word_maxlen

        # put each character into the matrix
        for j in range(word_len):
            char_len = len(X_char_ids[i][j])

            # in prediction, could see longer word
            if char_maxlen-char_len < 0:
                char_len = char_maxlen

            # left-padded with zeors
            vec = X_char_ids[i][j][:char_maxlen]
            X[i, word_maxlen-1-j, char_maxlen-char_len:] = vec

    return X





def create_data_matrix_Y(Y_seq_onehots, nb_samples, maxlen, num_classes):
    Y = np.zeros((nb_samples, maxlen, num_classes))

    for i in range(nb_samples):
        cur_len = len(Y_seq_onehots[i])
        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen
        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        Y[i, maxlen-cur_len:,:] = Y_seq_onehots[i][:maxlen]

    return Y


