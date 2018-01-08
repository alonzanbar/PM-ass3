import numpy as np
from utils import *

SUBJ_NUM = 9
LAMDA = 0.02
EPSILON = 0.1**10


cor,docs_size,vocab = read_data_lines("dataset/develop.txt")
docs_counts,docs_subj = zip(*cor)
documents_len = len(cor)
vocab_len = len(vocab)
w2id= {w:i for i,w in enumerate(vocab)}
Wti  = np.zeros((documents_len, 9))
Pik  = np.zeros((SUBJ_NUM, vocab_len))
alpha= np.zeros(SUBJ_NUM)
for t in range(documents_len):
    Wti[t][t % SUBJ_NUM]=1

    pass

def calculate_Pki():
    denom = np.zeros(SUBJ_NUM)
    for t,doc_size in enumerate(docs_size):
        for i in range(SUBJ_NUM):
            denom[i] += Wti[t][i] * doc_size
            for w, v in docs_counts[t].items():
                Pik[i][w2id[w]] += Wti[t][i] * v

    for t in range(documents_len):
        for w,v in docs_counts[t].items():
            for i in range(SUBJ_NUM):
                Pik[i][w2id[w]]= (Pik[i][w2id[w]] + LAMDA) / (denom[i] + vocab_len ** 2 * LAMDA)


def calculate_alpha():
    global alpha
    sums = np.sum(Wti,axis=0)
    for i in range(SUBJ_NUM):
        if sums[i] == 0:
            sums[i] == EPSILON
        alpha1= sums[i]/documents_len
        alpha[i] = alpha1
    alpha = alpha /sum(alpha)

if __name__ == "__main__":
    calculate_Pki()
    calculate_alpha()
    pass




