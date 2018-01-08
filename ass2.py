import sys
import math
from collections import defaultdict

import numpy as np

from utils import read_data_lines


def linston(freq,s, lam , x):
    return (freq + lam) / (s + lam *x)


def head_out(freq,Ph):
    return Ph[freq]

def prep(corpus,words_stat,smooth_f,  *args):
    pr=0
    for w in corpus:
        pr+= np.log(smooth_f(words_stat[w],*args))
    return 2**((-1.0/len(corpus))*pr)



def main(args):
    # if (len(args)<6):
    #     print("wrong args")
    #     return
    output= [None]*30
    DEV = read_data_lines(args[1]);


    vocab = set(DEV)
    output[0] = sys.argv[1]
    output[1] = sys.argv[2]
    INPUT_WORD = sys.argv[3]
    output[2] = INPUT_WORD
    output[3] = sys.argv[4]
    lang_vocab_len = 300000 #why ** 2 ?
    output[4] = lang_vocab_len
    output[5] = 1.0/lang_vocab_len
    dev_len = len(DEV)
    output[6]= dev_len
    train = DEV[0:int(math.ceil(dev_len*0.9))]
    validation = DEV[int(math.ceil(dev_len*0.9)):dev_len]
    count_T = defaultdict(int)
    count_V = defaultdict(int)
    for x in train:
        count_T[x]+=1
    for x in validation:
        count_V[x]+=1
    train_size = len(train)
    validation_size = len(validation)
    output[7] = validation_size
    output[8] = train_size
    train_voc_size =len(set(train))
    output[9] =train_voc_size
    freq_word = count_T[INPUT_WORD]
    output[10] = freq_word
    output[11]  = freq_word / len(train)
    freq_unseen = 0
    output[12]  = freq_unseen
    output[13] = linston(freq_word,train_size,0.1,train_voc_size)
    output[14] = linston(freq_unseen,train_size,0.1,train_voc_size)
    output[15] = prep(validation,count_T,linston,train_size,0.01,train_voc_size)
    output[16] = prep(validation,count_T,linston,train_size,0.10,train_voc_size)
    output[17] = prep(validation,count_T,linston,train_size,1.00,train_voc_size)
    # minimize lamda
    pr=[];
    lam = [l/1000.0 for l in xrange(1, 1000,20)]
    for l in lam:
       pr.append( prep(validation,count_T,linston,train_size,l,train_voc_size))
    best_lamda = lam[np.argmin(pr)]
    output[18]= lam[np.argmin(pr)]
    output[19] = np.min(pr)

    #Held on
    h_train = DEV[0:int(math.ceil(dev_len*0.5))]
    held_on = DEV[int(math.ceil(dev_len*0.5))+1:dev_len-1]
    h_train_size = len(h_train)
    held_on_size = len(held_on)
    count_T= defaultdict(int)
    count_H = defaultdict(int)
    for x in h_train:
        count_T[x]+=1
    for x in held_on:
        count_H[x]+=1

    count_to_words_T=defaultdict(list)
    for x in count_T.items():
        count_to_words_T[x[1]].append(x[0])
    count_to_words_H=defaultdict(list)
    for x in count_H.items():
        count_to_words_H[x[1]].append(x[0])
    Tr = defaultdict(int)
    Nr = defaultdict(int)
    #print("h_train_size %s , sum : %s" ) % (h_train_size , sum([k*len(v) for k,v in count_to_words.items()]))
    for x in count_to_words_T.items():
        Nr[x[0]] = len(x[1])
        for w1 in x[1]:
            Tr[x[0]]+=count_H[w1]
            count_H.pop(w1) # zeoring to elemetns words that are not in class 0 in oder to count class 0 easier
    Nr[0] = lang_vocab_len - len(vocab)
    Tr[0] = np.sum(count_H.values())
    #print("H: %s, sum: %s") %(len(held_on),sum(Tr.values()))

    Ph = {clas:1.0*cnt/(Nr[clas]*held_on_size) for  clas, cnt in Tr.items()}

    #freq_of_freqs = getfreq_of_freq(h_train,held_on)
    output[20] = h_train_size
    output[21] = held_on_size
    output[22] = Ph[count_T[INPUT_WORD]]
    output[23] = Ph[0]

    #test test
    TEST =read_data_lines(args[2]);
    count_Test = defaultdict(int)
    test_len = len(TEST)
    output[24] = test_len
    print(prep(TEST,count_T,linston,train_size,1.00,train_voc_size))
    print(prep(TEST,count_T,head_out,Ph))
    print ("\n".join("output %s: %s " % (i+1,o) for i,o in enumerate(output)))
    res = defaultdict(int)
    for i in range(0,9):
        fl = linston(i,train_size,best_lamda)*train_size
        fh = Ph[i]*h_train_size
        Ntr = Nr[i]
        Ttr = Tr[i]
        res[i] = tuple(fl,fh,Ntr,Ttr)
    output






if __name__ == "__main__":
    main(sys.argv)

