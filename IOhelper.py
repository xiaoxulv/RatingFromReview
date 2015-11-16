__author__ = 'Ariel'
import pandas as pd
import numpy as np
import json
import pickle
from scipy.sparse import csr_matrix



def readAsDF(file):
    # read data into a Dataframe
    json_list = []
    with open(file, 'r') as f:
        for line in f:
            json_list.append(json.loads(str(line)))
    print len(json_list)
    df = pd.DataFrame(json_list)
    return df

def stopword():
    # read stop words list
    stop_words = []
    with open('stopword.list', 'r') as f:
        for line in f:
            stop_words.append(line.strip())
    stop_words = set(stop_words)
    return stop_words

def storeModel(m, options, df, ifHash):
    # write matrix x and y to file
    if options == 2:
        output = 'dev'
    if options == 3:
        output = 'test'
    if options == 1:# train
        output = 'train'
        outfile = open(output + 'Y.pickle', 'wb')
        pickle.dump(list(df['stars']), outfile)

    if ifHash:
        output += 'Hash'
    outfile = open(output + '.pickle', 'w')
    pickle.dump(m, outfile)
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