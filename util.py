__author__ = 'Ariel'

import time
import json
import re
import heapq
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix

def preprocess(file, ifTrain, ifHash):
    size = 10000
    text_list, star_list = readJson(file, ifTrain)
    stopwords = readStopword()
    tokens = textProcess(text_list, stopwords)
    Dict = globalDict(tokens)
    top = selectTop(Dict, size)
    if ifHash:
        m = HashModel(tokens, top)
    else:
        m = BaseModel(tokens, top)
    return m, star_list

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

def textProcess(text_list, stopwords):
    tokens = []
    for i in xrange(len(text_list)):
        # remove punctuations and words with numbers
        text = re.sub('[^\w\s]|\w*\d\w*', '', text_list[i].lower())
        counts = Counter(text.split())
        for x in stopwords.intersection(counts):
            del counts[x]
        tokens.append(counts)
    return tokens

def globalDict(tokens):
    # Build global dictionary
    Dict = {}
    for x in tokens:
        for y in x:
            try:
                Dict[y] += 1
            except:
                Dict[y] = 1
    return Dict

def selectTop(Dict, size):
    return heapq.nlargest(size, Dict, key = Dict.get)

def locate(top):
    idx = range(len(top))
    toIdx = {}
    for x in idx:
        toIdx[top[x]] = x
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

