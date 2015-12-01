__author__ = 'Ariel'

import time
import json
import re
import math
import heapq
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix

def transformY(y):
    y = np.array(y)
    Y = np.zeros((y.shape[0], 5))
    for i,x in enumerate(y):
        Y[i][x-1] = 1
    return Y

def preprocess(file, ifTrain, ifHash, trainTop):
    if ifHash:
        size = 10000
    else:
        size = 1000
    text_list, star_list = readJson(file, ifTrain)
    stopwords = readStopword()
    tokens, Dict, _ = textProcess(text_list, stopwords, False)
    if ifTrain:
        top = selectTop(Dict, size)
    else:
        top = trainTop
    if ifHash:
        m = HashModel(tokens, top)
    else:
        m = BaseModel(tokens, top)
    return m, star_list, top

def custom_preprocess(file, ifTrain, ifHash, trainTop):
    if ifHash:
        size = 10000
    else:
        size = 1000
    text_list, star_list = readJson(file, False)
    stopwords = readStopword()
    tokens, termDict, docuDict = textProcess(text_list, stopwords, True)
    if ifTrain:
        top = selectTop(termDict, size)
    else:
        top = trainTop
    if ifHash:
        m = tfidfHashModel(tokens, top, docuDict)
    else:
        m = tfidfBaseModel(tokens, top, docuDict)
    return m, star_list, top

def readJson(file, ifTrain):
    # read json file
    text_list = []
    stars_list = []
    with open(file, 'r') as f:
        for line in f:
            cur = json.loads(str(line))
            text_list.append(cur['text'])
            if ifTrain:
                stars_list.append(cur['stars'])
    # print len(text_list)
    return text_list, stars_list

def readStopword():
    # read stop words list
    stopwords = set()
    with open('stopword.list', 'r') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


def textProcess(text_list, stopwords, ifCustom):
    tf_dict = {}
    df_dict = {}
    tokens = []
    for i in xrange(len(text_list)):
        # remove punctuations and words with numbers
        text = re.sub('[^\w\s]|\w*\d\w*', '', text_list[i].lower())
        counts = Counter(text.split())
        for x in stopwords.intersection(counts):
            del counts[x]
        for key, value in counts.iteritems():
            temp = tf_dict.get(key, 0) + value
            tf_dict[key] = temp
            if ifCustom:
                cur = df_dict.get(key, 0) + 1
                df_dict[key] = cur
        tokens.append(counts)
    return tokens, tf_dict, df_dict

def selectTop(Dict, size):
    top = heapq.nlargest(size, Dict, key = Dict.get)
    return top


def locate(top):
    idx = range(len(top))
    toIdx = dict(zip(top, idx))
    return toIdx

def BaseModel(tokens, top):
    toIdx = locate(top)
    top = set(top)
    row = []
    col = []
    data = []
    for i in xrange(len(tokens)):
        d = tokens[i]
        for x in d.keys():
            if x in top:
                row.append(i)
                col.append(toIdx[x])
                data.append(d[x])
    m = coo_matrix((data, (row, col)))
    m = csr_matrix(m)
    return m

def HashModel(tokens, top):
    top = set(top)
    row = []
    col = []
    data = []
    for i in xrange(len(tokens)):
        d = tokens[i]
        for x in d.keys():
            if x in top:
                row.append(i)
                col.append(hash(x)%1000)
                data.append(d[x])
    m = coo_matrix((data, (row, col)))
    m = csr_matrix(m)
    return m

def tfidfHashModel(tokens, top, docuDict):
    top = set(top)
    row = []
    col = []
    data = []
    N = len(tokens)
    for i in xrange(N):
        d = tokens[i]
        for x in d.keys():
            if x in top:
                row.append(i)
                col.append(hash(x)%1000)
                data.append(d[x] * math.log(1+N/(docuDict[x]+0.0)))
    m = coo_matrix((data, (row, col)), shape = (N, 1000))
    m = csr_matrix(m)
    return m

def tfidfBaseModel(tokens, top, docuDict):
    top = set(top)
    toIdx = locate(top)
    row = []
    col = []
    data = []
    N = len(tokens)
    for i in xrange(N):
        d = tokens[i]
        for x in d.keys():
            if x in top:
                row.append(i)
                col.append(toIdx[x])
                data.append(d[x] * math.log(1+N/(docuDict[x]+0.0)))
    m = coo_matrix((data, (row, col)), shape = (N, 1000))
    m = csr_matrix(m)
    return m

def writePred(t, s, output):
    with open(output, 'w') as f:
        for x in xrange(s.shape[0]):
            f.write(str(t[x]) + ' ' + str(s[x][0]) + '\n')
    return


# def globalDict(tokens):
#     # Build global dictionary
#     Dict = {}
#     for x in tokens:
#         for y in x:
#             try:
#                 Dict[y] += 1
#             except:
#                 Dict[y] = 1
#     return Dict

# def documentDict(Dict, tokens):
#     # Build document dictionary
#     docuDict = {}
#     for key in Dict.keys():
#         for x in tokens:
#             if key in x.keys():
#                 try:
#                     docuDict[key] += 1
#                 except:
#                     docuDict[key] = 1
#     return docuDict


# def readTop():
#     with open('test.pickle', 'r') as f1:
#         top = pickle.loads(f1.read())
#     return top