
import numpy as np
import os
import time
import random
import sys


from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.layers import Lambda, Masking


hierarchical_lstm = None


def train(X_word_ids, X_char_ids, Y_ids, tag2id, W=None, epochs=15, Y0=None):

    # gotta beef it up sometimes
    #X_char_ids = X_char_ids * 50
    #X_word_ids = X_word_ids * 50
    #Y_ids = Y_ids * 50
    '''
    n = len(X_word_ids)
    n = int(n * .60)
    X_word_ids = X_word_ids[:n]
    X_char_ids = X_char_ids[:n]
    Y_ids      =      Y_ids[:n]
    '''


    def matrix_max(list_of_lists):
        return max(map(max,list_of_lists))

    def matrix_len(list_of_lists):
        return max(map(len,list_of_lists))

    word_input_dim =         matrix_max(X_word_ids)  + 1
    char_input_dim = max(map(matrix_max,X_char_ids)) + 1

    #word_maxlen =         matrix_len(X_word_ids)
    #char_maxlen = max(map(matrix_len,X_char_ids))
    word_maxlen = 32
    char_maxlen = 16
    #word_maxlen = 4
    #char_maxlen = 4

    num_tags = len(tag2id)
    nb_samples = len(X_word_ids)

    # if we want to add the lower-level supervision
    if Y0 is not None:
        y0_tag2id = { w:i for i,w in enumerate(set(sum(Y0,[]))) }
        Y0_ids = [ [y0_tag2id[y] for y in y_seq] for y_seq in Y0 ]

        y0_num_tags  = len(y0_tag2id)
        Y0_seq_onehots = [to_categorical(y, nb_classes=y0_num_tags) for y in Y0_ids ]
        Y0 = create_data_matrix_Y(Y0_seq_onehots, nb_samples, word_maxlen, y0_num_tags)
        y0 = True
    else:
        y0 = False
        y0_num_tags = None

    hierarchical_lstm = create_model(word_input_dim, char_input_dim,
                                     word_maxlen   , char_maxlen   ,
                                     num_tags      , W             ,
                                     y0=(y0,y0_num_tags)           )

    # turn each id in Y_ids into a onehot vector
    Y_seq_onehots  = [to_categorical(y, nb_classes=num_tags) for y in Y_ids]

    # TODO - consider batching here if all data is too big for ram in dense matrix form

    # format X and Y data
    X_char = build_X_char_matrix(X_char_ids, nb_samples, word_maxlen,char_maxlen)
    X_word = build_X_word_matrix(X_word_ids, nb_samples, word_maxlen)

    Y  = create_data_matrix_Y( Y_seq_onehots, nb_samples, word_maxlen,    num_tags)

    print 
    print 'V_c:  ', char_input_dim
    print 'V_w:  ', word_input_dim
    print 'char: ', X_char.shape
    print 'word: ', X_word.shape
    print 

    #S = hierarchical_lstm.summary()
    #print S

    print 
    print 'training begin'
    print 

    batch_size = 512

    # optimize for more than one sequence of labels?
    if y0:
        target = [Y0,Y]
    else:
        target = Y

    #'''
    hierarchical_lstm.fit({'char': X_char, 'word':X_word},
                          target,
                          batch_size=batch_size ,
                          nb_epoch=epochs       ,
                          verbose=1              )
    #'''

    '''
    # break into minibatches
    perm = range(len(X_word))
    history = []
    n = len(X_word)
    for epoch in range(epochs):
        print '\tepoch: ', epoch

        # shuffle the data
        random.shuffle(perm)
        X_word = X_word[perm,:]
        X_char = X_char[perm,:]
        Y      =      Y[perm,:]

        # minibatches
        loss = 0
        for i in range((n/batch_size) + 1):
            # build minibatch
            start =  i   *batch_size
            end   = (i+1)*batch_size
            x_word = X_word[start:end,:]
            x_char = X_char[start:end,:,:]
            y      =      Y[start:end,:]

            if i==0:
                print '\t\tbatch: ', x_char.shape
                print '\t\t', 
            sys.stdout.write('.')

            # train on minibatch
            L = hierarchical_lstm.train_on_batch({'char':x_char,'word':x_word}, y)

            # weight during training
            weight = len(x_word)/float(n)
            loss += weight * L

        # monitor training loss
        print
        print '\t\tloss: ', loss
        history.append(loss)
    '''

    print 'training done'

    ######################################################################

    # temporary debugging-ness
    #hierarchical_lstm.load_weights('tmp_keras_weights')

    '''
    p = hierarchical_lstm.predict({'char': X_char, 'word':X_word},
                                  batch_size=batch_size)
    if y0:
        pred_prob = p[1]
    else:
        pred_prob = p
    #unoptimized_seq = p[0]
    #pred_prob       = p[1]
    #pred_prob       = p

    predictions = []
    for i in range(nb_samples):
        num_words = len(Y_ids[i])
        tags = pred_prob[i,word_maxlen-num_words:].argmax(axis=1)
        #print 'gold: ', np.array(Y_ids[i])
        #print 'pred: ', tags
        #print
        predictions.append(tags.tolist())
    #print '\n\n\n\n'

    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(predictions,Y_ids):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

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

    print '\n\n\n\n'

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

    print 'p: ',
    for p in precision: print '%4.2f' % p,
    print
    print 'r: ',
    for r in recall:    print '%4.2f' % r,
    print
    print 'f: ',
    for f in f1:        print '%4.2f' % f,
    print
    print 'avg-f1: ', np.mean(f1)
    print

    total_correct = sum( confusion[i,i] for i in range(num_tags) )
    total = confusion.sum()
    accuracy = total_correct / (total + 1e-9)
    print 'Accuracy: ', accuracy
    '''

    ######################################################################


    # accuracy crap

    # needs to return something pickle-able (so get binary serialized string)
    tmp_file = 'tmp_keras_weights-%d' % random.randint(0,10000)
    hierarchical_lstm.save_weights(tmp_file)
    with open(tmp_file, 'rb') as f:
        hierarchical_lstm_str = f.read()
    os.remove(tmp_file)

    # how many words in the vocabulary?
    W_shape = W.shape

    '''
    # Verify encoded correctly
    lstm = create_model(word_input_dim, char_input_dim,
                        word_maxlen   , char_maxlen   ,
                        num_tags      , W_shape=W_shape, 
                        y0=y0)

    p = hierarchical_lstm.predict({'char': X_char, 'word':X_word},
                                  batch_size=batch_size)
    exit()
    '''

    # return model back to cliner
    keras_model_tuple = (hierarchical_lstm_str,
                         word_input_dim, char_input_dim,
                         num_tags,
                         word_maxlen, char_maxlen,
                         W_shape, y0, y0_num_tags)
    return keras_model_tuple





def predict(keras_model_tuple, X_word_ids, X_char_ids, Y0=None):

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
    y0                    = keras_model_tuple[7]
    y0_num_tags           = keras_model_tuple[8]

    if not hierarchical_lstm:
        print '\t\tloading model from disk'

        # build LSTM
        hierarchical_lstm = create_model(word_input_dim, char_input_dim ,
                                         word_maxlen   , char_maxlen    ,
                                         num_tags      , W_shape=W_shape,
                                         y0=(y0,y0_num_tags))

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

    if y0:
        p = p[1]
    else:
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
                 W=None        , W_shape=None  ,
                 y0=(False,None)               ):

    # hyperparams
    char_emb_size  = 50
    c_seq_emb_size = 50
    wlstm1_size    = 150
    wlstm2_size    = 150

    # pretrained word embeddings
    if W is not None:
        word_emb_size  = W.shape[1]
        W_init = [W]
    elif W_shape is not None:
        word_emb_size  = W_shape[1]
        W_init = [np.random.rand(W_shape[0], W_shape[1])]
    else:
        word_emb_size  = 150
        #W = np.random.rand(word_input_dim,word_emb_size)
        W_init = None

    if y0[0]:
        num_tags_inner = y0[1]
        num_tags_outer = num_tags
    else:
        num_tags_inner = num_tags
        num_tags_outer = None

    # character-level LSTM encoder
    char_input = Input(shape=(char_maxlen,), dtype='int32')
    char_embedding = Embedding(output_dim=char_emb_size,
                               input_dim=char_input_dim,
                               input_length=char_maxlen,
                               mask_zero=True)(char_input)
    char_lstm_f    = LSTM(output_dim=c_seq_emb_size                  )(char_embedding)
    char_lstm_r    = LSTM(output_dim=c_seq_emb_size,go_backwards=True)(char_embedding)
    char_lstm_fr = merge([char_lstm_f,char_lstm_r], mode='concat')
    char_encoder_fr = Model(input=char_input, output=char_lstm_fr)

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
                               #mask_zero=False)(word_seqs)

    # combine char-level encoded states WITH word embeddings
    word_feats = merge([m_encoded_char_fr_states, word_embedding],
                       mode='concat',concat_axis=-1)
    #word_feats = encoded_char_fr_states
    #word_feats = word_embedding

    # Dropout
    #word_feats_d = TimeDistributed(Dropout(0.05))(word_feats)

    # word-level LSTM
    word_lstm_f1 = LSTM(output_dim=wlstm1_size, return_sequences=True
                                        )(word_feats)
                                        #)(word_feats_d)
    word_lstm_r1 = LSTM(output_dim=wlstm1_size, return_sequences=True,
                       go_backwards=True)(word_feats)
                       #go_backwards=True)(word_feats_d)
    word_lstm_fr1 = merge([word_lstm_f1, word_lstm_r1], mode='concat',concat_axis=-1)

    # Dropout
    #word_lstm_d1 = TimeDistributed(Dropout(0.05))(word_lstm_fr1)

    '''
    word_lstm_f2 = LSTM(output_dim=wlstm2_size, return_sequences=True
                                        )(word_lstm_fr1)
                                        #)(word_lstm_d1)
    word_lstm_r2 = LSTM(output_dim=wlstm2_size, return_sequences=True,
                       go_backwards=True)(word_lstm_fr1)
                       #go_backwards=True)(word_lstm_d1)
    word_lstm_fr2 = merge([word_lstm_f2, word_lstm_r2], mode='concat',concat_axis=-1)
    '''

    # Dropout
    #word_lstm_d2 = TimeDistributed(Dropout(0.05))(word_lstm_fr2)

    # Predict labels using the sequence of word encodings
    orig_pred = TimeDistributed( Dense(output_dim=num_tags_inner,
                                       activation='softmax' )  )(word_lstm_fr1)
                                       #activation='softmax' )  )(word_lstm_fr2)
                                       #activation='softmax' )  )(word_lstm_d2)

    model = Model( input=[char_seqs,word_seqs],
                   output=[orig_pred]  )

    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    # If thats it, then return right here (now second optimization)
    if not y0[0]:
        return model

    #seq1 = merge([word_feats, orig_pred], mode='concat',concat_axis=-1)
    seq1 = merge([word_feats, word_lstm_fr1], mode='concat',concat_axis=-1)

    # Sequence Optimizer
    seq_lstm_f1 = LSTM(output_dim=20,return_sequences=True
                                        )(seq1)
    seq_lstm_r1 = LSTM(output_dim=20,return_sequences=True,
                       go_backwards=True)(seq1)
    seq_lstm_fr1 = merge([seq_lstm_f1, seq_lstm_r1], mode='concat',concat_axis=-1)

    # Dropout
    #seq_lstm_d = TimeDistributed(Dropout(0.05))(seq_lstm_fr1)

    # Optimized label predictions
    opt_pred = TimeDistributed( Dense(output_dim=num_tags_outer,
                                      activation='softmax' )  )(seq_lstm_fr1)
                                      #activation='softmax' )  )(seq_lstm_d)

    model = Model( input=[char_seqs,word_seqs],
                   output=[orig_pred,opt_pred]  )

    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model








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





def create_data_matrix_Y(Y_seq_onehots, nb_samples, maxlen, nb_classes):
    Y = np.zeros((nb_samples, maxlen, nb_classes))

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


