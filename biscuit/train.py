######################################################################
#  CliNER - train.py                                                 #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: Build model for given training data.                     #
######################################################################

__author__ = 'Willie Boag'
__date__   = 'Aug. 15, 2016'

import os
import glob
import argparse
import sys
import cPickle as pickle

import tools
from model import GalenModel
from documents import Document

# base directory
CLINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(prog='cliner train')
    parser.add_argument("--txt",
        dest = "txt",
        help = ".txt files of discharge summaries"
    )
    parser.add_argument("--annotations",
        dest = "con",
        help = "concept files for annotations of the .txt files",
    )
    parser.add_argument("--model",
        dest = "model",
        help = "Path to the model that should be stored",
    )
    parser.add_argument("--format",
        dest = "format",
        help = "Data format ( i2b2 )"
    )
    parser.add_argument("--log",
        dest = "log",
        help = "Path to the log file for training info",
        default = os.path.join(CLINER_DIR, 'models', 'train.log')
    )

    parser.add_argument("--no-use-crf",
        dest = "no_use_crf",
        action = 'store_true',
        help = "Disable CRF layer on top",
        default = False
    )
    parser.add_argument("--no-use-lstm",
        dest = "no_use_lstm",
        action = 'store_true',
        help = "Disable word lstm",
        default = False
    )
    parser.add_argument("--no-use-char-lstm",
        dest = "no_use_char_lstm",
        action = 'store_true',
        help = "Disable char lstm",
        default = False
    )
    parser.add_argument("--embeddings",
        dest = "embeddings",
        help = "Path to read word embeddings from",
        default = os.path.join(CLINER_DIR, 'models', 'glove.6B.100d.txt')
    )
    parser.add_argument("--dropout",
        dest = "dropout",
        help = "Dropout probability p",
        default = 0.5
    )
    # Parse the command line arguments
    args = parser.parse_args()

    # Error check: Ensure that file paths are specified
    if not args.txt:
        print >>sys.stderr, '\n\tError: Must provide text files'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.con:
        print >>sys.stderr, '\n\tError: Must provide annotations for text files'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.model:
        print >>sys.stderr, '\n\tError: Must provide valid path to store model'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    modeldir = os.path.dirname(args.model)
    if (not os.path.exists(modeldir)) and (modeldir != ''):
        print >>sys.stderr, '\n\tError: dir does not exist: %s' % modeldir
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)

    # data format
    if not args.format:
        print '\n\tERROR: must provide "format" argument\n'
        exit()

    # Must specify output format
    if args.format not in ['i2b2']:
        print >>sys.stderr, '\n\tError: Must specify output format'
        print >>sys.stderr,   '\tAvailable formats: i2b2'
        print >>sys.stderr, ''
        exit(1)

    # collect all hyperparameter arguments
    hyperparams = {}
    hyperparams['use_crf']       = not args.no_use_crf
    hyperparams['use_lstm']      = not args.no_use_lstm
    hyperparams['use_char_lstm'] = not args.no_use_char_lstm
    hyperparams['embeddings']    = args.embeddings
    hyperparams['dropout']       = float(args.dropout)

    # A list of text and concept file paths
    txt_files = glob.glob(args.txt)
    con_files = glob.glob(args.con)

    # Collect training data file paths
    txt_files_map = tools.map_files(txt_files)
    con_files_map = tools.map_files(con_files)

    training_list = []
    for k in txt_files_map:
        if k in con_files_map:
            training_list.append((txt_files_map[k], con_files_map[k]))

    #opt = args.opt.split('_')

    # Train the model
    train(training_list, args.model,args.format,logfile=args.log,hyperparams=hyperparams)



#def train(training_list, model_path, format, logfile=None, opt=["concept"]):
def train(training_list, model_path, format, logfile=None, hyperparams={}):
    # Read the data into a Document object
    docs = []
    for txt, con in training_list:
        #try:
            doc_tmp = Document(txt, con)
            docs.append(doc_tmp)
        #except Exception, e:
        #    exit( '\n\tWARNING: Document Exception - %s\n\n' % str(e) )

    # file names
    if not docs:
        print 'Error: Cannot train on 0 files. Terminating train.'
        exit(1)

    # Create a Machine Learning model
    model = GalenModel()

    # Train the model using the Document's data
    #model.fit_from_documents(docs, opt)
    model.fit_from_documents(docs, hyperparams=hyperparams)

    # Pickle dump
    print '\nserializing model to %s\n' % model_path
    model.serialize(model_path, logfile=logfile)


if __name__ == '__main__':
    main()
