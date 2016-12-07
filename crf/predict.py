######################################################################
#  CliNER - predict.py                                               #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Use trained model to predict concept labels for data.    #
######################################################################


__author__ = 'Willie Boag'
__date__   = 'Oct. 5, 2014'


import os
import sys
import glob
import argparse
import tools
import cPickle as pickle

from model import CRF
from documents import Document


def main():

    parser = argparse.ArgumentParser(prog='crf predict')
    parser.add_argument("--txt",
        dest = "txt",
        help = ".txt files of discharge summaries"
    )
    parser.add_argument("--out",
        dest = "output",
        help = "The directory to write the output",
    )
    parser.add_argument("--model",
        dest = "model",
        help = "The model to use for prediction",
    )
    parser.add_argument("--format",
        dest = "format",
        help = "Data format ( con )"
    )
    args = parser.parse_args()

    # Error check: Ensure that file paths are specified
    if not args.txt:
        print >>sys.stderr, '\n\tError: Must provide text files\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.output:
        print >>sys.stderr, '\n\tError: Must provide output directory\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.model:
        print >>sys.stderr, '\n\tError: Must provide path to model\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not os.path.exists(args.model):
        print >>sys.stderr, '\n\tError: CRF does not exist: %s\n' % args.model
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)


    # Parse arguments
    files = glob.glob(args.txt)
    tools.mkpath(args.output)

    if args.format:
        format = args.format
    else:
        print '\n\tERROR: must provide "format" argument\n'
        exit()


    # Predict
    predict(files, args.model, args.output, format=format)



def predict(files, model_path, output_dir, format):
    # Must specify output format
    if format not in ['i2b2']:
        print >>sys.stderr, '\n\tError: Must specify output format'
        print >>sys.stderr,   '\tAvailable formats: i2b2'
        print >>sys.stderr, ''
        exit(1)

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Tell user if not predicting
    if not files:
        print >>sys.stderr, "\n\tYou did not supply any input files\n"
        exit(1)

    # For each file, predict concept labels
    n = len(files)
    for i,txt in enumerate(sorted(files)):
        note = Document(txt)

        fname = os.path.splitext(os.path.basename(txt))[0] + '.con'
        out_path = os.path.join(output_dir, fname)
        '''
        if os.path.exists(out_path):
            #print '\tWARNING: prediction file already exists (%s)' % out_path
            continue
        '''


        print '-' * 30
        print '\n\t%d of %d' % (i+1,n)
        print '\t', txt, '\n'


        # Predict concept labels
        labels = model.predict_classes_from_document(note)

        # Get predictions in proper format
        output = note.write(labels)


        # Output the concept predictions
        print '\n\nwriting to: ', out_path
        with open(out_path, 'w') as f:
            print >>f, output
        print



if __name__ == '__main__':
    main()
