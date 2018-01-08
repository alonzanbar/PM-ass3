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
        vocab|=set(words)
        corpus.append((c,y))
        docs_size.append(sum(c.values()))
        line_num += 2
    return corpus, docs_size,vocab

if __name__=='main':
    cor = read_data_lines("dataset/develop.txt")
pass





