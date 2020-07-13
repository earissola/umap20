#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd

__author__ = "Esteban Rissola"
__credits__ = ["Esteban Rissola"]
__version__ = "1.0.1"
__maintainer__ = "Esteban Rissola"
__email__ = "esteban.andres.rissola@usi.ch"


class NRC_toolbox(object):
    def __init__(self):
        super(NRC_toolbox, self).__init__()
        self.emotion_lexicon = {}
        self.affect_lexicon = {}
        self.emotions_idx = {'anger': 0, 'anticipation': 1, 'disgust': 2,
                             'fear': 3, 'joy': 4, 'negative': 5, 'positive': 6,
                             'sadness': 7, 'surprise': 8, 'trust': 9}
        self.affect_dim_idx = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
        self.load_emotion_lexicon()
        self.load_affect_lexicon()

    def load_emotion_lexicon(self):
        emotion_lexicon_path = 'NRC-Emotion-Lexicon-Wordlevel.txt'
        # IMPORTANT: Words are repeated #
        with open(emotion_lexicon_path, 'rt') as fp:
            lines = fp.readlines()
            for idx in (range(0, len(lines), 10)):
                for row in lines[idx:idx + 10]:
                    w, e, f = row.strip().split()
                    if not w in self.emotion_lexicon:
                        self.emotion_lexicon[w] = np.zeros(
                            len(self.emotions_idx), dtype=np.int32)
                    self.emotion_lexicon[w][self.emotions_idx[e]] = int(f)

    def load_affect_lexicon(self):
        affect_lexicon_path = 'NRC-AffectIntensity-Lexicon.txt'
        with open(affect_lexicon_path, 'rt') as fp:
            for line in fp:
                word, score, affect_dim = line.strip().split()
                if word not in self.affect_lexicon:
                    self.affect_lexicon[word] = [None, None, None, None]
                idx = self.affect_dim_idx[affect_dim]
                self.affect_lexicon[word][idx] = (affect_dim, float(score))

    def print_emotions(self):
        print(self.emotions_idx.keys())

    def get_emotions_frequency(self, words):
        emotions_acc = np.zeros(len(self.emotions_idx), dtype=np.int32)
        for w in words:
            if w in self.emotion_lexicon:
                emotions_acc += self.emotion_lexicon[w]
        return emotions_acc

    def get_sadness_score(self, words):
        sadness_scores = []
        sadness_scores_avg = 0.0
        for w in words:
            if w in self.affect_lexicon:
                idx = self.affect_dim_idx['sadness']
                if (self.affect_lexicon[w][idx] != None) and (self.affect_lexicon[w][idx][0] == 'sadness'):
                    sadness_scores.append(self.affect_lexicon[w][idx][1])
        if len(sadness_scores) > 0:
            sadness_scores_avg = reduce(
                lambda x, y: x + y, sadness_scores) / len(sadness_scores)
        return sadness_scores_avg
