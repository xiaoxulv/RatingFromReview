__author__ = 'Ariel'

import numpy as np
import time
import pandas as pd
import re
import json


json_list = []
start_time = time.time()
with open('yelp_reviews_train.json', 'r') as f:
    for line in f:
        #print line
        json_list.append(json.loads(str(line)))
        #d = pd.read_json(line)
print len(json_list)
df = pd.DataFrame(json_list)
print 'time: %ss' % (time.time()-start_time)

df['text'] = df['text'].map(lambda x: x.lower()) # faster than str.lower()
#df['text'] = df["text"].str.lower()
print 'time: %ss' % (time.time()-start_time)

#df['text'] = df['text'].str.replace('[^\w\s]|\w*\d\w*',' ')
df['text'].replace({'[^\w\s]|\w*\d\w*': ''}, regex = True, inplace = True)
# df['text'] = df['text'].apply(raw_string)
# df['text'] = df['text'].apply(replace)
print 'time: %ss' % (time.time()-start_time)

df['tokens'] = df['text'].str.split()
print 'time: %ss' % (time.time()-start_time)

stop_words = []
with open('stopword.list', 'r') as f:
    for line in f:
        stop_words.append(line.strip())
stop_words = set(stop_words)

#df['tokens_new'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop_words])

def remove(l,stop):
    l = set(l)
    for x in stop.intersection(l):
        if x in l:
            l.remove(x)
    return l
df['tokens_new'] = df['tokens'].apply(remove, args = (stop_words,))

print 'time: %ss' % (time.time()-start_time)

df['tokens_new'].to_csv('tokens.txt', index = False)
print 'time: %ss' % (time.time()-start_time)
# def replace(s):
#     return re.sub('[^\w\s]|\w*\d\w*', s, ' ')
# def raw_string(s):
#     if isinstance(s, str):
#         s = s.encode('string-escape')
#     elif isinstance(s, unicode):
#         s = s.encode('unicode-escape')
#     return s