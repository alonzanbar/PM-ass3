import operator
import numpy as np
from utils import *

SUBJ_NUM = 9
LAMDA = 0.2
EPSILON = 0.1**10
K=10

cor,docs_size,vocab = read_data_lines("dataset/develop.txt")
Ntk, docs_subj = zip(*cor)
documents_len = len(cor)
vocab_len = len(vocab)
w2id= {w:i for i,w in enumerate(vocab)}
Wti  = np.zeros((documents_len, SUBJ_NUM))
Zti = np.zeros((documents_len,SUBJ_NUM))
Mt = np.zeros(documents_len)
Pik  = np.zeros((SUBJ_NUM, vocab_len))
alpha= np.zeros(SUBJ_NUM)
lang_vocab_len = 300000  # len(vocabulary) ** 2 #vacebulary length


def calculate_Pki():
    denom = np.zeros(SUBJ_NUM)
    temp = np.zeros((SUBJ_NUM,vocab_len))
    for t,doc_size in enumerate(docs_size):
        for i in range(SUBJ_NUM):
            denom[i] += Wti[t][i] * doc_size
            for w, v in Ntk[t].items():
                temp[i][w2id[w]] += Wti[t][i] * v
    for i in range(SUBJ_NUM):
        for w in w2id:
            Pik[i][w2id[w]]= (temp[i][w2id[w]] + LAMDA) / (denom[i] +  lang_vocab_len * LAMDA)


def calculate_alpha():
    global alpha
    sums = np.sum(Wti,axis=0)
    for i in range(SUBJ_NUM):
        if sums[i] == 0:
            sums[i] = EPSILON
        alpha1= sums[i]/documents_len
        alpha[i] = alpha1
    alpha = alpha /sum(alpha)


def calculate_Wti():
    for t in range(documents_len):
        z = Zti[t]
        m= Mt[t]
        sum_z= sum([np.e**(i-m) for i in z if (i-m)>=-K])
        for i in range(len(z)):
            if (z[i]-m)<-K:
                Wti[t][i]=0
            else:
                Wti[t][i] = np.e**(z[i]-m) / sum_z

def caclculate_Zti():
    for t in range(documents_len):
        z = np.zeros(SUBJ_NUM)
        for i in range(SUBJ_NUM):
            for w,v in Ntk[t].items():
                z[i]+= v*np.log(Pik[i][w2id[w]])
            Zti[t][i] = np.log(alpha[i])+z[i]

        Mt[t] = max(Zti[t])

    pass

def calculate_loss():
    loss=0
    for t in range(documents_len):
        tl=0
        for i in range(SUBJ_NUM):
            if (Zti[t][i] - Mt[t])>= -K:
                tl+=np.e**(Zti[t][i] - Mt[t])
            else:
                pass
        loss+=Mt[t] + np.log(tl)
    return loss

def accuracy():
    categuries =defaultdict(lambda: defaultdict(int))
    clusters = defaultdict(list)
    for t in range(documents_len):
        c = np.argmax(Wti[t])
        clusters[c].append(t)
        for ci in docs_subj[t]:
            categuries[c][ci] += 1


    finalCat = {}
    for c in range(SUBJ_NUM):
        if len(categuries[c])>0:
            finalCat[c] = (max(collections.Counter(categuries[c]).items(), key=operator.itemgetter(1))[0])

    correct = 0
    for c,vec in clusters.items():
        for t in vec:
            if finalCat[c] in docs_subj[t]:
                correct+= 1

    print("accuracy -> " + str(correct/len(Wti)))

if __name__ == "__main__":

    for t in range(documents_len):
        Wti[t][t % SUBJ_NUM] = 1

    calculate_Pki()
    calculate_alpha()
    caclculate_Zti()
    accuracy()
    for it in range(10):
        # E step
        calculate_Wti()

        # M step
        calculate_Pki()
        calculate_alpha()
        caclculate_Zti()

        # loss
        loss = calculate_loss()
        print(loss)
        accuracy()

    pass




