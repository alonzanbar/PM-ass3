import operator
import numpy as np
from utils import *
import matplotlib.pyplot as plt

SUBJ_NUM = 9
LAMDA = 0.3
EPSILON = 0.1**5
K=10
STOP_PERCENT = 0.0001

cor,docs_size,vocab,words_size = read_data_lines("dataset/develop.txt")
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
#print ("vocab_len "+str(vocab_len))

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
            Pik[i][w2id[w]]= (temp[i][w2id[w]] + LAMDA) / (denom[i] +  vocab_len * LAMDA)
        #print(Pik[i])
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    #print("PiK :")
    #print(np.sum(Pik,axis=1))



def calculate_alpha():
    global alpha
    sums = np.sum(Wti,axis=0)
    for i in range(SUBJ_NUM):
        if sums[i] == 0:
            sums[i] = EPSILON
        alpha1= sums[i]/documents_len
        alpha[i] = alpha1
        #print(alpha1)
    alpha = alpha /sum(alpha)


def calculate_Wti():
    for t in range(documents_len):
        z = Zti[t]
        m= Mt[t]
        sum_z= sum([np.e**(i-m) for i in z ])
        for i in range(len(z)):
            if (z[i]-m)<-K:
                Wti[t][i]=0
            else:
                Wti[t][i] = np.e**(z[i]-m) / sum_z
    #print("WtI : ")
    #print(np.sum(Wti, axis=0))

# def calculate_Wti():
#     for t in range(len(Ntk)):
#         z = np.zeros(SUBJ_NUM)
#         sm=0
#         for i in range(SUBJ_NUM):
#             skm = 0
#             for k,v in Ntk[t].items():
#                 skm += v * np.log(Pik[i][w2id[k]])
#             z[i] = np.log(alpha[i])+skm
#             sm += np.e**z[i]
#
#         m = np.max(z)
#         smr = 0
#         for i in range(SUBJ_NUM):
#             if ((z[i]-m)>=-K):
#                 smr +=np.e**(z[i]-m)
#
#         for i in range(SUBJ_NUM):
#             if (z[i] - m < -K):
#                 Wti[t][i]=0
#             else:
#                 Wti[t][i] = np.e**(z[i]-m)/smr


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

def calculate_preplexity(loss,words):
    prep = 2.0 ** ((-1 / float(words)) * loss)
    return prep

def accuracy(is_last_iteration):
    categuries =defaultdict(lambda: defaultdict(int))
    clusters = defaultdict(list)
    for t in range(documents_len):
        c = np.argmax(Wti[t])
        clusters[c].append(t)
        for ci in docs_subj[t]:
            categuries[c][ci] += 1


    finalCat = [None]*SUBJ_NUM
    for c in range(SUBJ_NUM):
        if len(categuries[c])>0:
            finalCat[c] = (max(collections.Counter(categuries[c]).items(), key=operator.itemgetter(1))[0])

    correct = 0
    for c,vec in clusters.items():
        for t in vec:
            if finalCat[c] in docs_subj[t]:
                correct+= 1

    print("accuracy -> " + str(correct/len(Wti)))

    if(not is_last_iteration):
        #print(finalCat)
        #print("_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_")
        topics = ReadTopics("dataset/topics.txt")
        matrix = np.zeros((len(topics), len(topics)))

        for c, vec in clusters.items():
            for t in vec:
                if finalCat[c] in docs_subj[t]:
                    correct += 1


        row = 0
        for row_t in topics:
            col = 0
            for col_t in topics:
                if col_t in finalCat:
                    index = finalCat.index(col_t)
                    matrix[row,col] = categuries[row][finalCat[index]]
                else:
                    matrix[row,col] = 0
                col+=1

            #print(matrix[row])
            #print(categuries[row])
            #print("************************")
            row += 1


        #print(1)

if __name__ == "__main__":
    lossIteration = []
    preplexityIteration = []
    last_loss = 0
    loss_percent = 1
    it = 1

    for t in range(documents_len):
        Wti[t][t % SUBJ_NUM] = 1

    calculate_Pki()
    calculate_alpha()
    caclculate_Zti()
    accuracy(loss_percent > STOP_PERCENT)


    while loss_percent > STOP_PERCENT:
    #for it in range(50):
        # E step
        calculate_Wti()

        # M step
        calculate_Pki()
        calculate_alpha()
        caclculate_Zti()
        #print(alpha)

        # loss
        loss = calculate_loss()
        #perplexity
        preplexity = calculate_preplexity(loss,words_size)

        if(it > 1):
            loss_percent = np.abs((loss - last_loss) / last_loss)
        print(loss_percent)
        print("itrration " + str(it) + " the loss is "+str(loss))
        print("itrration " + str(it) + " the perplexity is " + str(preplexity))
        accuracy(loss_percent > STOP_PERCENT)
        last_loss = loss
        it+=1
        lossIteration.append(last_loss)
        preplexityIteration.append(preplexity)

    iteration = []
    for i in range(len(lossIteration)):
        iteration.append(i+1)
    plt.plot(iteration,lossIteration)
    plt.ylabel('some numbers')
    plt.show()
    plt.plot(iteration, preplexityIteration)
    plt.ylabel('some numbers')
    plt.show()
    pass




