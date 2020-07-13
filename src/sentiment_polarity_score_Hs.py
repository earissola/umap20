#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Heuristic: Sentiment Polarity Score (Hs). Research has shown that the sentiment 
polarity score of a post can be linked with the emotions evoked by a piece text.
We hypothesise that when this value is negative (i.e., below zero) it can be a 
good indicator ofdistress or unhappiness, especially when the posts are written 
by users experiencing depression.
'''

import argparse
import sys
from csv import reader, register_dialect, writer
from glob import glob
from operator import itemgetter
from os import getcwd, listdir, makedirs
from os.path import basename, exists, isdir, join, splitext
from statistics import mean, median, stdev

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
SADNESS_SCORE_TRHS = 0.1


def generate_training_set_positive(corpus_pp_path, thresholds):
    training_set = []
    nrc_toolbox = NRC_toolbox()
    pp_wc = []
    with open(corpus_pp_path, 'rt') as fp:
        r = reader(fp, dialect='tab')
        # Discard Header #
        next(r)
        for doc_id, text, document, _posting_date in r:
            doc_id = int(doc_id)
            words = document.split()

            if len(words) >= MIN_PP_WC_POSITIVE:
                tb = TextBlob(text)
                polarity = tb.sentiment.polarity
                emotions = nrc_toolbox.get_emotions_frequency(words)
                # sel_emotions = np.take(emotions, [2, 5, 7])
                # 2: Disgust, 5: Negative, 7: Sadness
                sel_emotions = np.take(emotions, [2, 7, 5])
                sadness_score = nrc_toolbox.get_sadness_score(words)
                wc = len(words)
                # Filter by 'polarity score' #
                if polarity < thresholds['polarity_tb']:
                    training_set.append(
                        (doc_id, document, polarity, sel_emotions, 
                        sadness_score, wc))
    # Sort by 'polarity score' #
    training_set = sorted(training_set, key=itemgetter(2))
    with open('hs_positive.txt', 'w') as fp:
        for doc_id, document, polarity, sel_emotions, sadness_score, wc in training_set:
            # Filter by 'binary_emotions_occurence' #
            # 0: Disgust, 1: Sadness
            disgust = 1 if sel_emotions[0] > 0 else 0
            sadness = 1 if sel_emotions[1] > 0 else 0
            # Require both of them (disgust and sadness) to be present #
            if ((disgust > 0) or (sadness > 0)) and sadness_score > SADNESS_SCORE_TRHS:
                pp_wc.append(len(document.split()))
                output_str = '%d\t%s\t%.4f\t%s\t%.4f\t%d\n'
                fp.write(output_str % (doc_id, document, polarity, 
                                       np.array2string(sel_emotions), 
                                       sadness_score, wc))
        pp_wc_mean = mean(pp_wc)
        pp_wc_median = median(pp_wc)
        pp_wc_std = stdev(pp_wc)
        stats = (pp_wc_mean, pp_wc_median, pp_wc_std)
        print('Average Document Lenght (WC)')
        print('Mean: %.4f || Median: %.4f || Stdev: %.4f (Positive)' % stats)

def generate_training_set_control(corpus_path):
    training_set = []
    nrc_toolbox = NRC_toolbox()
    pp_wc = []
    with open(corpus_path, 'rt') as fp:
        r = reader(fp, dialect='tab')
        # Discard Header #
        next(r)
        for doc_id, text, document, _posting_date in r:
            doc_id = int(doc_id)
            words = document.split()
            if len(words) >= MIN_PP_WC_CONTROL:
                tb = TextBlob(text)
                polarity = tb.sentiment.polarity
                training_set.append((doc_id, document, polarity))
        shuffled = np.array(range(0, len(training_set)))
        np.random.seed(SEED)
        np.random.shuffle(shuffled)
        with open('hs_control.txt', 'wt') as fp:
            for idx in shuffled:
                doc_id, document, polarity = training_set[idx]
                words = document.split()
                emotions = nrc_toolbox.get_emotions_frequency(words)
                # sel_emotions = np.take(emotions, [2, 5, 7])
                # 2: Disgust, 5: Negative, 7:Sadness
                sel_emotions = np.take(emotions, [2, 7, 5])
                sadness_score = nrc_toolbox.get_sadness_score(words)
                wc = len(words)
                pp_wc.append(len(words))
                output_str = '%d\t%s\t%.4f\t%.4f\t%s\t%.4f\t%d\n'
                fp.write(output_str % (doc_id, document, polarity, 
                                       np.array2string(sel_emotions), 
                                       sadness_score, wc))
        pp_wc_mean = mean(pp_wc)
        pp_wc_median = median(pp_wc)
        pp_wc_std = stdev(pp_wc)
        stats = (pp_wc_mean, pp_wc_median, pp_wc_std)
        print('Mean: %.4f || Median: %.4f || Stdev: %.4f (Control)' % stats)


def main():
    description_message = 'Heuristic: Sentiment Polarity Score (Hs)'
    parser = argparse.ArgumentParser(description=description_message)
    help_msgs = []
    help_msgs.append('corpus_pp path (positive)')
    help_msgs.append('corpus_pp path (control)')

    parser.add_argument('corpus_pp_positive_path', help=help_msgs[0])
    parser.add_argument('corpus_pp_control_path', help=help_msgs[1])

    # Arguments parsing #
    args = parser.parse_args()

    # Random useful variables definition #
    register_dialect('tab', delimiter='\t')
    thresholds = {'polarity_tb': 0.0}

    generate_training_set_positive(args.corpus_pp_positive_path, thresholds)
    generate_training_set_control(args.corpus_pp_control_path)

    return 0


if __name__ == '__main__':
    main()
