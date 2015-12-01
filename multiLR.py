__author__ = 'Ariel'


import time
import math
import util
import eval
import numpy as np
import matplotlib.pyplot as plt
from random import randint


# def SGD(X, Y):
#
#     start_time = time.time()
#
#     W = np.ones([5, X.shape[1]]) * 1.0/ X.shape[1]
#     nabla_list = []
#     lambdada = 0.05
#     step = 0.001
#     iter = 0
#     while iter < 100000:
#
#         r = randint(0, X.shape[0]-1)
#         sumover = 0
#         for j in xrange(5):
#             #print W[j]*(X[j].transpose())
#             sumover += math.exp(W[j]*(X[r].transpose()))
#         softmax = np.exp(W * (X[r].transpose()))/sumover
#         temp = Y[r].reshape([5, 1]) - softmax
#         nabla = temp * X[r] - lambdada * W
#         # adaptive learning rate
#         step = 10.0/(1000+iter)
#         W = W + step * nabla
#
#         #train prediction
#         if iter%1000 == 0:
#             Sumover = 0
#             for j in xrange(5):
#                 Sumover += np.exp(W[j]*(X.transpose()))
#             distri = np.exp(W * (X.transpose()))/Sumover
#
#             t = np.argmax(distri, axis=0)
#             t = t + 1
#             print eval.accuracy(t, y)
#            # nabla_list.append(np.linalg.norm(nabla))
#
#         iter += 1
#
#     print 'time: %ss' % (time.time()-start_time)
#     #plt.plot(nabla_list)
#     #plt.show()
#     return W


def BSGD(X, y):
    start_time = time.time()
    Y = util.transformY(y)
    W = np.ones([5, X.shape[1]]) * 1.0 / X.shape[1]
    nabla_list = []
    lambdada = 0.05
    step = 0.001
    iter = 0
    batch_size = 500
    chunk_list = chunks(range(X.shape[0]), batch_size)
    round = int(math.ceil(X.shape[0]/(batch_size+0.0)))

    while iter < 5023:
        # iteratively update
        r = chunk_list[iter%round]
        sumover = np.zeros(X[r].shape[0]).reshape([1, X[r].shape[0]])
        for j in xrange(5):
            sumover += np.exp(W[j]*(X[r].transpose()))
        softmax = np.exp(W * (X[r].transpose()))/sumover
        temp = Y[r].T - softmax
        nabla = temp * X[r] - lambdada * W

        # adaptive learning rate
        # step = 10.0/(1000+iter)# adaptive learning rate

        W = W + step * nabla
        # train prediction
        if iter%round == 0:
            nabla_list.append(step*np.linalg.norm(nabla))
            t, s = predict(W, X)
            print eval.accuracy(t, y)
            print eval.eval(t, s, y)

        iter += 1
        if iter == 5023:
            t, s = predict(W, X)
            print eval.accuracy(t, y)
            print eval.eval(t, s, y)

    print 'time: %ss' % (time.time()-start_time)
    plt.plot(nabla_list)
    plt.show()

    return W


def predict(W, x):
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
    return t, s

def chunks(l, n):
    res = []
    for i in xrange(0, len(l), n):
        res.append(l[i:i+n])
    return res

