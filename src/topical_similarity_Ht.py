#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Heuristic: Topical Similarity (Ht). We define Ht as a heuristic in which less 
useful posts in D+ are filtered out based on their topical similarity with a 
depression taxonomy. 
'''

import argparse
import sys
from csv import reader, register_dialect, writer
from glob import glob
from operator import itemgetter
from os import getcwd, listdir, makedirs
from os.path import basename, exists, isdir, join, splitext

import numpy as np
import pandas as pd
from textblob import TextBlob

from lib.homemade import NRC_toolbox

__author__ = "Esteban Rissola"
__credits__ = ["Esteban Rissola"]
__version__ = "1.0.1"
__maintainer__ = "Esteban Rissola"
__email__ = "esteban.andres.rissola@usi.ch"

SEED = 10710
MIN_PP_WC_POSITIVE = 1
MIN_PP_WC_CONTROL = 1

def generate_training_set_positive(corpus_pp_path, sim_scores):
    training_set = []
    with open(corpus_pp_path, 'rt') as fp:
        r = reader(fp, dialect='tab')
        # Discard Header #
        next(r)
        for doc_id, _text, document, _posting_date in r:
            doc_id = int(doc_id)
            words = document.split()

            if len(words) >= MIN_PP_WC_POSITIVE:
                doc_sim_score = sim_scores[doc_id]
                wc = len(words)
                training_set.append((doc_id, document, wc, doc_sim_score))

    # Sort by similarity score #
    training_set = sorted(training_set, key=itemgetter(7), reverse=True)
    with open('ht_positive.txt', 'wt') as fp:
        for doc_id, document, wc, doc_sim_score in training_set:
            output_str = '%d\t%s\t%d\t%.4f\t\n'
            fp.write(output_str % (doc_id, document, wc, doc_sim_score))


def generate_training_set_control(corpus_path, sim_scores):
    training_set = []
    with open(corpus_path, 'rt') as fp:
        r = reader(fp, dialect='tab')
        # Discard Header #
        next(r)
        for doc_id, _text, document, _posting_date in r:
            doc_id = int(doc_id)
            words = document.split()
            if len(words) >= MIN_PP_WC_CONTROL:
                doc_sim_score = sim_scores[doc_id]
                training_set.append((doc_id, document, doc_sim_score))
        shuffled = np.array(range(0, len(training_set)))
        np.random.seed(SEED)
        np.random.shuffle(shuffled)
        with open('ht_control.txt', 'wt') as fp:
            for idx in shuffled:
                doc_id, document, doc_sim_score = training_set[idx]
                words = document.split()
                wc = len(words)
                output_str = '%d\t%s\t%.4f\t%s\t%.4f\t%d\n'
                fp.write(output_str % (doc_id, document, wc, doc_sim_score))

def load_set_sim_scores(filepath):
    sim_scores = np.loadtxt(filepath, delimiter='\t', dtype=np.float64)
    return sim_scores

def main():
    description_message = 'Heuristic: Topical Similarity (Ht)'
    parser = argparse.ArgumentParser(description=description_message)
    help_msgs = []
    help_msgs.append('corpus_pp path (positive)')
    help_msgs.append('corpus_pp path (control)')
    help_msgs.append('sim_scores path')
    
    parser.add_argument('corpus_pp_positive_path', help=help_msgs[0])
    parser.add_argument('corpus_pp_control_path', help=help_msgs[1])
    parser.add_argument('sim_scores_path', help=help_msgs[2])

    # Arguments parsing #
    args = parser.parse_args()

    # Random useful variables definition #
    register_dialect('tab', delimiter='\t')

    sim_scores = np.load(args.sim_scores_path)
    generate_training_set_positive(args.corpus_pp_positive_path, sim_scores)
    generate_training_set_control(args.corpus_pp_control_path, sim_scores)

    return 0


if __name__ == '__main__':
    main()
