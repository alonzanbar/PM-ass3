# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import collections
import random
from collections import defaultdict

def linston(freq,s, lam , x):
    return (freq + lam) / (s + lam *x)

def read_data_lines(file):
    corpus=[]
    docs_size=[]
    words_count = collections.Counter()
    vocab = set()
    with open(file, 'r') as f:
        lines = f.readlines()
        line_num = 0

    while line_num < len(lines):
        line = lines[line_num]
        y = line.split()[2:]
        y[-1] = y[-1][:-1]
        line_num+=2
        line = lines[line_num]
        words = line.split()
        c = collections.Counter(words)
        words_count+=c
        corpus.append((c,y))
        line_num += 2
    words_count = {x:words_count[x] for x in words_count if words_count[x]>3}
    for c,y in corpus:
        for w,v in list(c.items()):
            if w not in words_count:
                c.pop(w)
        docs_size.append(sum(c.values()))
        vocab=set(words_count.keys())
        words = sum(words_count.values())
    return corpus, docs_size,vocab,words

def ReadTopics(file):
    topics = []
    with open(file, 'r') as f:
        for l in f:
            if l == '\n':
                continue
            else:
                topics.append(l.rstrip())
    return topics

if __name__=='__main__':
    cor = read_data_lines("dataset/develop.txt")
pass





