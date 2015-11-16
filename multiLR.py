__author__ = 'Ariel'

import pickle
import time
import numpy as np
from random import randint
import math
import matplotlib.pyplot as plt
import pprint
import eval

def SGD(X, Y):
    start_time = time.time()

    W = np.random.rand(5, X.shape[1])
    nabla_list = []
    lambdada = 0.05
    step = 0.001
    iter = 0
    while iter < 5000:
        r = randint(0, X.shape[0])
        sumover = 0
        for j in xrange(5):
            #print W[j]*(X[j].transpose())
            sumover += math.exp(W[j]*(X[r].transpose()))
        softmax = np.exp(W * (X[r].transpose()))/sumover
        temp = Y[r].reshape([5, 1]) - softmax
        nabla = temp * X[r] - lambdada * W
        nabla_list.append(np.linalg.norm(nabla))
        #step = 10/(1000+iter)# adaptive learning rate
        W = W + step * nabla
        #print np.sqrt(np.sum(np.square(step*nabla)))
        #print iter
        iter += 1

    print 'time: %ss' % (time.time()-start_time)
    plt.plot(nabla_list)
    plt.show()
    return W


def BSGD(X, Y):
    with open('trainY.pickle', 'rb') as f2:
        y = pickle.loads(f2.read())
    y = np.array(y)

    start_time = time.time()

    W = np.random.rand(5, X.shape[1])
    nabla_list = []
    lambdada = 0.05
    step = 0.001
    iter = 0
    chunk_list = chunks(range(X.shape[0]), 100)

    cursor = 0
    while iter < 150000:
        #print iter

        r = chunk_list[cursor%12554]

        sumover = np.zeros(X[r].shape[0]).reshape([1, X[r].shape[0]])
        for j in xrange(5):
            #print W[j]*(X[j].transpose())
            sumover += np.exp(W[j]*(X[r].transpose()))
        softmax = np.exp(W * (X[r].transpose()))/sumover
        temp = Y[r].T - softmax
        nabla = temp * X[r] - lambdada * W
        if cursor%12544 == 0:
            nabla_list.append(step*np.linalg.norm(nabla))
        #step = 10/(1000+iter)# adaptive learning rate
        W = W + step * nabla
        #print np.sqrt(np.sum(np.square(step*nabla)))
        #print iter

        #train prediction
        if cursor%12554 == 0:
            Sumover = 0
            for j in xrange(5):
                Sumover += np.exp(W[j]*(X.transpose()))
            distri = np.exp(W * (X.transpose()))/Sumover

            t = np.argmax(distri, axis=0)
            t = t + 1
            print eval.accuracy(t, y)

        iter += 1
        cursor += 1

    print 'time: %ss' % (time.time()-start_time)
    plt.plot(nabla_list)
    plt.show()

    return W


def predict(W, ifdev):
    if ifdev:
        with open('dev.pickle', 'r') as f:
            x = pickle.loads(f.read())
    else:
        with open('test.pickle', 'r') as f:
            x = pickle.loads(f.read())

    Sumover = 0
    for j in xrange(5):
        Sumover += np.exp(W[j]*(x.transpose()))
    distri = np.exp(W * (x.transpose()))/Sumover

    # hard prediction
    t = np.argmax(distri, axis=0)
    t = t + 1
    # soft prediction
    s = np.zeros([distri.shape[1],1])
    for x in xrange(distri.shape[1]):
        s[x] = sum(i*j for i,j in zip(range(1,6), distri[:,x]))

    with open('v1.txt', 'w') as f:
        for x in xrange(s.shape[0]):
            f.write(str(t[x]) + ' ' + str(s[x][0]) + '\n')

    return

def chunks(l, n):
    res = []
    for i in xrange(0, len(l), n):
        res.append(l[i:i+n])
    return res

