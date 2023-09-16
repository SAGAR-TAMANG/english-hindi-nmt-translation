import pandas as pd
import numpy as np
import math
import nltk
import re 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from autocorrect import spell

# LOADING THE DATA 

import codecs 

with codecs.open('data/master-db.txt', encoding = 'utf-8') as f:
    data = f.read()

row = data.split('\n')
regex = re.compile('[%s]' % re.escape(string.punctuation))

# PREPROCESSING THE DATA

word_eng_dic = {}
word_hi_dic = {}

for i in row:
    s = i.split('\t')

    if len(s) == 2:
        s1 = regex.sub('', s[0])

        words = word_tokenize(s1)

        for word_tokens in words:
            w = word_tokens.lower()
            w = spell(w)
            word_eng_dic[w] = []
        
        s1 = regex.sub('', s[1])

        words = word_tokenize(s1)

        for word_tokens in words:
            
            if ord('।') == ord(word_tokens[-1]):
                print(word_tokens)
                
                word_hi_dic[word_tokens[:-1]]=[]
                print(word_tokens[:0])
            
            else:
                word_hi_dic[word_tokens]=[]

print(len(word_hi_dic))

# READING 50 DIM GLOVE EMBEDDINGS FROM FILE FOR ENGLISH VOCAB

count = 0
cnt = 0

with open('data/glove.6B.50d.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        values = line.split()

        word_weights = np.asarray(values[1:], dtype=np.float32)

        if values[0] in word_eng_dic:
            print("exist ", cnt)
            cnt += 1
            word_eng_dic[s[0]] = word_weights
        
        count += 1

    print(count, " ", cnt)


# FINDING THE WORDS NOT IN EMBEDDING FILE 

mispell = []
cnt = 0

for i in word_eng_dic.keys():
    if i in word_eng_dic:  # Check if the key exists in the dictionary
        try:
            a = word_eng_dic[i].shape
        except AttributeError:
            print(f"Word '{i}' has no embedding (AttributeError)")
            mispell.append(i)
            cnt += 1
    else:
        print(f"Word '{i}' is not in the embedding dictionary")
        mispell.append(i)
        cnt += 1

print("Total words without embeddings:", cnt)
