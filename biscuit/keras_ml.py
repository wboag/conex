
import numpy as np
import os
import time
import random
import sys
import StringIO

import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.layers.crf import ChainCRF
from keras.layers.wrappers import TimeDistributed
from keras.layers import Masking
from keras.callbacks import EarlyStopping

from tools import matrix_max
from documents import id2tag, labels as tag2id

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

    #train_Y = create_data_matrix_Y(train_Y_seq_onehots, nb_train_samples, 
    #                              word_maxlen, num_tags)
    train_Y = create_id_matrix_Y(train_Y_seq_onehots, nb_train_samples, 
                                  word_maxlen, num_tags)

    val_X_char = build_X_char_matrix(val_X_char_ids, nb_val_samples, 
                                       word_maxlen, char_maxlen)
    val_X_word = build_X_word_matrix(val_X_word_ids, nb_val_samples, word_maxlen)

    #val_Y = create_data_matrix_Y(val_Y_seq_onehots, nb_val_samples, 
    #                              word_maxlen, num_tags)
    val_Y = create_id_matrix_Y(val_Y_seq_onehots, nb_val_samples, 
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

    batch_size = 128

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

    # dropout
    p = 0.5

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
    char_emb = Embedding(output_dim=char_emb_size,
                         input_dim=char_input_dim,
                         input_length=char_maxlen,
                         mask_zero=True)(char_input)
    char_lstm_f  = LSTM(units=c_seq_emb_size                  ,
                        dropout=p, recurrent_dropout=p)(char_emb)
    char_lstm_r  = LSTM(units=c_seq_emb_size,go_backwards=True,
                        dropout=p, recurrent_dropout=p)(char_emb)
    char_lstm    = Concatenate()([char_lstm_f,char_lstm_r])

    # TODO: verify that go_backwards does the right thing (using toy example)
    #       MAY need to reverse the sequence in keras code
    pass

    char_encoder = Model(inputs=char_input, outputs=char_lstm)

    # apply char-level encoder to every char sequence (word)
    char_seqs = Input(shape=(word_maxlen,char_maxlen),dtype='int32',name='char')
    encoded_char_states = TimeDistributed(char_encoder)(char_seqs)
    m_encoded_char_states = Masking(0.0)(encoded_char_states)

    # apply embeddings layer to every word
    word_seqs = Input(shape=(word_maxlen,), dtype='int32', name='word')
    word_embedding = Embedding(output_dim=word_emb_size,
                               input_dim=word_input_dim,
                               input_length=word_maxlen,
                               weights=W_init,
                               mask_zero=True)(word_seqs)

    # combine char-level encoded states WITH word embeddings
    word_feats = Concatenate(axis=-1)([m_encoded_char_states, word_embedding])
    #word_feats = encoded_char_fr_states
    #word_feats = word_embedding

    # word-level LSTM
    '''
    word_lstm_f1 = LSTM(units=wlstm1_size, return_sequences=True,
                        dropout=p, recurrent_dropout=p, 
                                         )(word_feats)
    word_lstm_r1 = LSTM(units=wlstm1_size, return_sequences=True,
                        dropout=p, recurrent_dropout=p, 
                        go_backwards=True)(word_feats)
    word_lstm = Concatenate(axis=-1)([word_lstm_f1, word_lstm_r1])
    '''
    word_lstm = Bidirectional( LSTM(units=wlstm1_size, return_sequences=True,
                                   dropout=p, recurrent_dropout=p)           )(word_feats)

    # Predict labels using the sequence of word encodings
    orig_pred = TimeDistributed( Dense(units=num_tags_inner,
                                       activation='softmax' )  )(word_lstm)
    #orig_pred = TimeDistributed(Dense(units=num_tags_inner))(word_lstm)

    # TODO: crf layer
    crf = ChainCRF()
    #crf_output = crf(word_embedding)
    crf_output = crf(orig_pred)
    #crf_output = orig_pred

    #model = Model( inputs=[word_seqs],
    model = Model( inputs=[char_seqs,word_seqs],
                   outputs=[crf_output]        )
                   
    print
    print 'compiling model'
    start = time.clock()
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.compile(loss=crf.sparse_loss, optimizer='adam')
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
    iob_predictions = []
    con_predictions = []
    for i in range(nb_samples):
        num_words = len(Y_ids[i])
        iob_tags = pred[i,word_maxlen-num_words:].argmax(axis=1)
        concept_tags = [ id2tag[p][2:] for p in iob_tags ]
        con_predictions.append([ c if c else 'O' for c in concept_tags ])
        iob_predictions.append(iob_tags.tolist())
    concepts = list(set([ con[2:] if con!='O' else 'O' for con in tag2id ]))

    # confusion matrix
    iob_confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(iob_predictions,Y_ids):
        for y,p in zip(yseq, tags):
            iob_confusion[p,y] += 1

    num_concepts = len(concepts)
    concept_confusion = np.zeros( (num_concepts,num_concepts) )
    con2id = { c:i for i,c in enumerate(concepts) }
    concept_Y_empty = [ [id2tag[y][2:] for y in y_line] for y_line in Y_ids ]
    concept_Y = [ [c if c else 'O' for c in C] for C in concept_Y_empty ]
    for tags,yseq in zip(con_predictions,concept_Y):
        for y,p in zip(yseq, tags):
            p_ind = con2id[p]
            y_ind = con2id[y]
            concept_confusion[p_ind,y_ind] += 1

    # IOB confusion matrix
    out = StringIO.StringIO()
    out.write('\n\n%s IOB\n\n' % label)
    out.write(' '*7)
    for i in range(num_tags):
        out.write('%4d ' % i)
    out.write(' (gold)\n')
    for i in range(num_tags):
        out.write('%2d    ' % i)
        for j in range(num_tags):
            out.write('%4d ' % iob_confusion[i][j])
        out.write('\n')
    out.write('(pred)\n\n\n')
    iob_conf_str = out.getvalue()
    out.close()

    iob_precision = np.zeros(num_tags)
    iob_recall    = np.zeros(num_tags)
    iob_f1        = np.zeros(num_tags)

    for i in range(num_tags):
        correct    =     iob_confusion[i,i]
        num_pred   = sum(iob_confusion[i,:])
        num_actual = sum(iob_confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        iob_precision[i] = p
        iob_recall[i]    = r
        iob_f1[i]        = (2*p*r) / (p + r + 1e-9)

    # concept confusion matrix
    out = StringIO.StringIO()
    out.write('\n\n%s concepts\n\n' % label)
    out.write(' '*14)
    for c in concepts:
        out.write('%-10s ' % c)
    out.write(' (gold)\n')
    for i in range(num_concepts):
        out.write('%10s    ' % concepts[i])
        for j in range(num_concepts):
            out.write('%-10d ' % concept_confusion[i][j])
        out.write('\n')
    out.write('(pred)\n\n\n')
    con_conf_str = out.getvalue()
    out.close()

    con_precision = np.zeros(num_concepts)
    con_recall    = np.zeros(num_concepts)
    con_f1        = np.zeros(num_concepts)

    for i in range(num_concepts):
        correct    =     concept_confusion[i,i]
        num_pred   = sum(concept_confusion[i,:])
        num_actual = sum(concept_confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        con_precision[i] = p
        con_recall[i]    = r
        con_f1[i]        = (2*p*r) / (p + r + 1e-9)

    scores = {}

    scores['con_conf'     ] = con_conf_str
    scores['con_precision'] = con_precision
    scores['con_recall'   ] = con_recall
    scores['con_f1'       ] = con_f1

    scores['iob_conf'     ] = iob_conf_str
    scores['iob_precision'] = iob_precision
    scores['iob_recall'   ] = iob_recall
    scores['iob_f1'       ] = iob_f1

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



def create_id_matrix_Y(Y_seq_onehots, nb_samples, maxlen, num_classes):
    Y = np.zeros((nb_samples, maxlen,1))

    for i in range(nb_samples):
        cur_len = len(Y_seq_onehots[i])
        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen
        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        lst = [ [it] for it in Y_seq_onehots[i][:maxlen].argmax(axis=1) ]
        vec = np.matrix(lst)
        #Y[i, maxlen-cur_len:,:] = [Y_seq_onehots[i][:maxlen].argmax(axis=1)]
        Y[i, maxlen-cur_len:,:] = vec

    return Y


