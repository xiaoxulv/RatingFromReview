__author__ = 'Ariel'

import numpy as np
import time
import pandas as pd
import re
import json
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
import pickle


def generateModel(input_file, options):
    start_time = time.time()
    # load data into a Dataframe
    json_list = []
    with open(input_file, 'r') as f:
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
    if options == 2:
        output = 'dev'
    if options == 3:
        output = 'test'
    if options == 1:# train
        output = 'train'
        outfile = open(output + 'Y.pickle', 'wb')
        pickle.dump(list(df['stars']), outfile)

    outfile = open(output + '.pickle', 'w')
    pickle.dump(m, outfile)

    print 'time: %ss' % (time.time()-start_time)

    # data exploration part
    # for x in xrange(9):
    #     print words_select[x], Dict[words_select[x]]
    # print df.groupby(['stars']).count()

    return

def loadModel():
    # load in training model
    with open('train.pickle', 'r') as f1:
        X = pickle.loads(f1.read())
    X = csr_matrix(X)

    with open('trainY.pickle', 'rb') as f2:
        y = pickle.loads(f2.read())
    y = np.array(y)
    Y = np.zeros((y.shape[0], 5))
    for i,x in enumerate(y):
        Y[i][x-1] = 1
    return X, Y