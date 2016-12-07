######################################################################
#  CliNER - tools.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: General purpose tools                                    #
######################################################################


import os
import errno
import string
import re
import cPickle as pickle


#############################################################
#  files
#############################################################

def map_files(files):
    """Maps a list of files to basename -> path."""
    output = {}
    for f in files: #pylint: disable=invalid-name
        basename = os.path.splitext(os.path.basename(f))[0]
        output[basename] = f
    return output


def mkpath(path):
    """Alias for mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


#############################################################
#  text pre-processing
#############################################################

def clean_text(text):
    return ''.join(map(lambda x: x if (x in string.printable) else '@', text))


def normalize_tokens(toks):

    # normalize dosages (icluding 8mg -> mg)

    # replace number tokens
    def num_normalize(w):
        return '__num__' if re.search('\d', w) else w
    toks = map(num_normalize, toks)

    return toks


#############################################################
#  manipulating list-of-lists
#############################################################

def flatten(list_of_lists):
    '''
    flatten()

    Purpose: Given a list of lists, flatten one level deep

    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of objects (AKA flattened one level)

    >>> flatten([['a','b','c'],['d','e'],['f','g','h']])
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    '''
    return sum(list_of_lists, [])



def save_list_structure(list_of_lists):
    '''
    save_list_structure()

    Purpose: Given a list of lists, save way to recover structure from flattended

    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists

    >>> save_list_structure([['a','b','c'],['d','e'],['f','g','h']])
    [3, 5, 8]
    '''

    offsets = [ len(sublist) for sublist in list_of_lists ]
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]

    return offsets




def reconstruct_list(flat_list, offsets):

    '''
    save_list_structure()

    Purpose: This undoes a list flattening. Uses value from save_list_structure()

    @param flat_list. <list> of objects
    @param offsets    <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    @return           <list-of-lists> of objects (the original structure)

    >>> reconstruct_list(['a','b','c','d','e','f','g','h'], [3,5,8])
    [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]
    '''

    return [ flat_list[i:j] for i, j in zip([0] + offsets, offsets)]




#############################################################
#  serialization to disc
#############################################################

def load_pickled_obj(path_to_pickled_obj):
    data = None
    with open(path_to_pickled_obj, "rb") as f:
        data = f.read()
    return pickle.loads(data)


def pickle_dump(obj, path_to_obj):
    # NOTE: highest priority makes loading TRAINED models slow
    with open(path_to_obj, 'wb') as f:
        pickle.dump(obj, f, -1)



#############################################################
#  prose v nonprose
#############################################################


def is_prose_sentence(sentence):
    assert type(sentence) == type([]), 'is_prose_sentence() must take list arg'
    if sentence == []:
        return False
    #elif sentence[-1] == '.' or sentence[-1] == '?':
    elif sentence[-1] == '?':
        return True
    elif sentence[-1] == ':':
        return False
    elif len(sentence) <= 5:
        return False
    elif is_at_least_half_nonprose(sentence):
        return True
    else:
        return False



def is_at_least_half_nonprose(sentence):
    count = len(filter(is_prose_word, sentence))
    if count >= len(sentence)/2:
        return True
    else:
        return False



def is_prose_word(word):
    # Punctuation
    for punc in string.punctuation:
        if punc in word:
            return False
    # Digit
    if re.match('\d', word):
        return False
    # All uppercase
    if word == word.upper():
        return False
    # Else
    return True




def prose_partition(tokenized_sents, labels=None):
    prose_sents     = []
    nonprose_sents  = []
    prose_labels    = []
    nonprose_labels = []

    # partition the sents & labels into EITHER prose OR nonprose groups
    for i in range(len(tokenized_sents)):
        if is_prose_sentence(tokenized_sents[i]):
            prose_sents.append(tokenized_sents[i])
            if labels:
                prose_labels.append(labels[i])
        else:
            nonprose_sents.append(tokenized_sents[i])
            if labels:
                nonprose_labels.append(labels[i])

    # group data appropriately (note, labels might not be provided)
    if labels:
        prose    = (   prose_sents,    prose_labels)
        nonprose = (nonprose_sents, nonprose_labels)
    else:
        prose    = (   prose_sents, None)
        nonprose = (nonprose_sents, None)

    return prose, nonprose




