__author__ = 'Ariel'

import numpy as np
import time
import pandas as pd
import re
import json
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
import pickle
import IOhelper

def generateModel(input_file, options, ifHash):
    start_time = time.time()

    df = IOhelper.readAsDF(input_file)
    stop_words = IOhelper.stopword()
    print 'time: %ss' % (time.time()-start_time)

    df_new, words = textPreprocess(df, stop_words)
    print 'time: %ss' % (time.time()-start_time)
    Dict = globalDict(words)
    print 'time: %ss' % (time.time()-start_time)

    if ifHash:
        m = HashModel(Dict, 10000, df_new)
    else:
        m = BaseModel(Dict, 1000, df_new)
    print 'time: %ss' % (time.time()-start_time)

    IOhelper.storeModel(m, options, df_new, ifHash)
    print 'time: %ss' % (time.time()-start_time)

    return


def textPreprocess(df, stop_words):
    # text preprocess and tokenize
    tokens_dict = []
    words = []
    for i in xrange(df.shape[0]):
        # remove punctuations and words with numbers
        text = re.sub('[^\w\s]|\w*\d\w*', '', df['text'][i].lower())
        counts = Counter(text.split())
        for x in stop_words.intersection(counts):
            del counts[x]
        tokens_dict.append(counts)
        words.append(counts.elements())
    df['tokens'] = tokens_dict
    return df, words

def globalDict(words):
    # Build global dictionary
    Dict = {}
    for x in words:
        for y in x:
            try:
                Dict[y] += 1
            except:
                Dict[y] = 1
    return Dict

def BaseModel(Dict, size, df):
    # baseline model of size
    words_select = sorted(Dict, key = Dict.__getitem__, reverse = True)[:size]
    set_words = set(words_select)
    # Build model matrix of x
    row = []
    col = []
    data = []
    for i in xrange(df.shape[0]):
        d = df['tokens'][i]
        for x in d.keys():
            if x in set_words:
                row.append(i)
                col.append(words_select.index(x))
                data.append(d[x])
    m  = coo_matrix((data, (row, col)))
    # data exploration part
    # for x in xrange(9):
    #     print words_select[x], Dict[words_select[x]]
    # print df.groupby(['stars']).count()
    return m

def HashModel(Dict, size, df):
    # feature hashing model
    words_select = sorted(Dict, key = Dict.__getitem__, reverse = True)[:size]
    set_words = set(words_select)
     # Build model matrix of x
    row = []
    col = []
    data = []
    for i in xrange(df.shape[0]):
        d = df['tokens'][i]
        for x in d.keys():
            if x in set_words:
                row.append(i)
                col.append(hash(x)%1000)
                data.append(d[x])
    m  = coo_matrix((data, (row, col)))
    return m