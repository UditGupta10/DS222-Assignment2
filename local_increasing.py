import pickle
import time
import os
import re
import math
#import matplotlib.pyplot as plt
learning_rate = 0.001
lamda = 0.05      #Regularization param
n = 1           #n-grams

epochs = 1
plot_train_acc = []
plot_test_acc = []
plot_train_loss = []

wt = dict()
last = dict()
idf = dict()
tfidf = dict()

st = set()
classes = set()
tf_idf()
print("Learning rate is constant")

for cnt in range(0,epochs):
    train_start = time.time()
    train(cnt, lr * (2.0**cnt))
    train_end = time.time()

    test_start = time.time()
    test(cnt)
    test_end = time.time()

print("Train time per epoch = {} seconds".format(train_end - train_start))
print("Test time per epoch = {} seconds".format(test_end - test_start))
with open("plot_train_acc.txt", "wb") as fp:   #Pickling
    pickle.dump(plot_train_acc, fp)

with open("plot_train_loss.txt", "wb") as fp:   #Pickling
    pickle.dump(plot_train_loss, fp)

with open("plot_test_acc.txt", "wb") as fp:   #Pickling
    pickle.dump(plot_test_acc, fp)
# plt.plot([1,2,3], plot_train_acc)
# plt.show()
#print(len(st))


def train(cnt,learning_rate):
    for cnt1 in range(1,2):
        itr = 0
        c = 0
        t = 0
        error = 0
        itr_classes = 0
        f = open("/scratch/ds222-2017/assignment-1/archive/backup/DBPedia.verysmall/verysmall_train.txt",'r')
        for i in f.readlines():
            itr += 1
            tokens = i.split(' ')
            max_prob = 0
            max_class = ""
            key = tokens[0].split(',')
            new_tokens = [''.join(tokens[i:i+n]) for i in range(1,len(tokens) - (n - 1))]
            for clas in classes:                            
                if clas in key:
                    y = 1
                else:
                    y = 0
                x = 0                          
                for w in new_tokens:
                    wc = re.sub(r'[^\w+\st+]','',w)
                    x += wt[wc][clas] * tfidf[wc][clas]
                #print(x,end = ' ')
                p = sigmoid(x)          
                if p > max_prob:                
                    max_prob = p
                    max_class = clas
                #print(p)
                if y == 1 and p != 0:
                    error += -math.log(p)
                elif p != 1:
                    error += -math.log(1-p)
                for w in new_tokens:
                    wc = re.sub(r'[^\w+\st+]','',w)
                    #if wt[wc][clas] > 0:
                    wt[wc][clas] = wt[wc][clas] + (y - p) * learning_rate * tfidf[wc][clas] - 2 * learning_rate * lamda * wt[wc][clas]      #update weights
            if max_class in key:
                c += 1
            t += 1

        f.close()
        plot_train_acc.append(float(c)/t*100)
        plot_train_loss.append((error*1.0)/itr)
        
def test(cnt):
    itr = 0
    c = 0
    t = 0
    f = open("/scratch/ds222-2017/assignment-1/archive/backup/DBPedia.verysmall/verysmall_test.txt",'r')
    for i in f.readlines():
        itr += 1
        #print(itr)
        tokens = i.split(' ')
        key = tokens[0].split(',')
        new_tokens = [''.join(tokens[i:i+n]) for i in range(1,len(tokens) - (n - 1))]
        y = 1
        l = list()
        max_prob = 0
        max_class = ""
        for clas in classes:                             
            x = 0                           
            for w in new_tokens:
                wc = re.sub(r'[^\w+\st+]','',w)
                if wc in st:
                    x += wt[wc][clas] * tfidf[wc][clas]
            
            p = sigmoid(x)       
            if p > max_prob:                
                max_prob = p
                max_class = clas
        if max_class in key:
            c += 1
        t += 1


    plot_test_acc.append((float(c) / t) * 100)
    
    f.close()

def tf_idf():
        itr = 0
        f = open("/scratch/ds222-2017/assignment-1/archive/backup/DBPedia.verysmall/verysmall_train.txt",'r')  #open file
        #f = open('data','r')
        for i in f.readlines():
            itr += 1
            tokens = i.split(' ')   
            key = tokens[0].split(',')      
            new_tokens = [''.join(tokens[i:i+n]) for i in range(1,len(tokens) - (n - 1))]          
            for k in key:
                classes.add(k)         
            for w in new_tokens:            
                wc = re.sub(r'[^\w+\st+]','',w)      
                if wc not in last:    
                    st.add(wc)
                    last[wc] = dict()
                for k in key:
                    if k not in last[wc]:     
                        last[wc][k] = 1
                    else:
                        last[wc][k] += 1   
        
        for w in st:          
            idf[w] = math.log(itr / len(last[w]))
            wt[w] = dict()
        
        for w in st:             #initialize tfidf and weights
            tfidf[w] = dict()
            for clas in classes:
                tfidf[w][clas] = 0
                wt[w][clas] = 0
        
        for w in last:
            i = idf[w]
            for c in last[w]:
                tfidf[w][c] += math.log(last[w][c] + 1) * i             #calculate tfidf

        f.close()

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
