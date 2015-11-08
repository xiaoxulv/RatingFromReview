__author__ = 'Ariel'

import numpy as np
import time
import pandas as pd
import re
import json
from collections import Counter
from scipy.sparse import coo_matrix
import pickle


def process(file):
    start_time = time.time()
    # load data into a Dataframe
    json_list = []
    with open(file, 'r') as f:
        for line in f:
            json_list.append(json.loads(str(line)))
    print len(json_list)
    df = pd.DataFrame(json_list)
    # read in stop words list
    stop_words = []
    with open('stopword.list', 'r') as f:
        for line in f:
            stop_words.append(line.strip())
    stop_words = set(stop_words)
    print 'time: %ss' % (time.time()-start_time)

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
    print 'time: %ss' % (time.time()-start_time)

    # Build overall dictionary
    Dict = {}
    for x in words:
        for y in x:
            try:
                Dict[y] += 1
            except:
                Dict[y] = 1
    print 'time: %ss' % (time.time()-start_time)

    # Model selection
    words_select = sorted(Dict, key = Dict.__getitem__, reverse = True)[:1000]
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
    print 'time: %ss' % (time.time()-start_time)

    # write matrix x and y to file
    outfile = open('train.pickle', 'w')
    pickle.dump(m, outfile)

    outfile = open('trainY.pickle', 'wb')
    pickle.dump(list(df['stars']), outfile)
    print 'time: %ss' % (time.time()-start_time)

    # data exploration part
    # for x in xrange(9):
    #     print words_select[x], Dict[words_select[x]]
    # print df.groupby(['stars']).count()

    return