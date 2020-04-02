import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

class Process(object):
    """docstring for Process."""
    dataframe = pd.read_csv("New_data.csv")
    garbage = set(stopwords.words('english')+list(punctuation))
    finaldata=[]
    train1 = []
    train2=[]
    train3=[]
    train4=[]
    train5=[]
    accu=[]
    def processTweets(self):
        newt = []
        self.garbage = list(self.garbage)
        with open("Output.txt", "w") as text_file:
            for i in range(len(self.garbage)):
                text_file.write(self.garbage[i])
        for i in range(len(self.dataframe)):
            self.finaldata.append([self.processT2(self.dataframe.iloc[i,0].lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
           .replace(";", "").replace(":", "").replace("*", "").replace("\"","")
           .replace("(", "").replace(")", "").replace("-","").replace("_","")
           .replace("/", "").replace("$","").replace("#","").replace("0","").replace("2","").replace("1","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","").replace("8","").replace("9","")),int(self.dataframe.iloc[i,1])])
           # self.finaldata.append([self.processT2(self.dataframe.iloc[i,0].lower()),int(self.dataframe.iloc[i,1])])
        # print(self.finaldata)
        np.random.shuffle(self.finaldata)
    def processT2(self,sentence):
        # n_sentence = sentence.split()
        n_sentence = word_tokenize(sentence)
        # print(n_sentence)   #If not NLTK use split
        # if n_sentence!=sentence:
        #     print(sentence,n_sentence)
        return [word for word in n_sentence if word not in self.garbage]
    def k_fold_cross(self,train,test):
        words_in_cl0=0
        words_in_cl1=0
        vocab = {}

        for i in range(len(train)):
            for j in train[i][0]:
                if j in vocab:
                    continue
                else :
                    vocab[j]=1
        print(len(vocab))
        for i in range(len(train)):
            if train[i][1]==1:
                words_in_cl1 += len(train[i][0])
            else :
                words_in_cl0 += len(train[i][0])
        dict1 = {}
        dict0 = {}
        print(words_in_cl0)
        print(words_in_cl1)
        for i in range(len(train)):
            for j in train[i][0]:
                if train[i][1]==1:
                    if j in dict1:
                        dict1[j]+=1
                    else:
                        dict1[j]=1
                else :
                    if j in dict0:
                        dict0[j]+=1
                    else:
                        dict0[j]=1

        a0 = 0
        a1 = 0
        a0 = np.log(words_in_cl0/len(train))
        a1 = np.log(words_in_cl1/len(train))
        pred=[]
        cnt=0
        cnt1=0
        for i in range(len(test)):
            a2=a0
            a3=a1
            for j in test[i][0]:
                if j in dict0:
                    a2+=np.log(dict0[j]/words_in_cl0)
                    cnt+=1
                else:
                    a2+=np.log(1/(words_in_cl0+len(vocab)))
                    cnt1+=1
                if j in dict1:
                    a3+=np.log(dict1[j]/words_in_cl1)
                    cnt+=1
                else:
                    a3+=np.log(1/(words_in_cl1+len(vocab)))
                    cnt1+=1
            if(a2>a3):
                pred.append(0)
            else :
                pred.append(1)
        aa=0
        print(cnt,cnt1)
        for i in range(len(pred)):
            if int(pred[i])==test[i][1]:
                aa+=1
        print(aa/200)
        self.accu.append(aa/200)
        return

if __name__ == '__main__':
    pro = Process()
    pro.processTweets()
    pro.train1 = pro.finaldata[:200]
    pro.train2 = pro.finaldata[200:400]
    pro.train3 = pro.finaldata[400:600]
    pro.train4 = pro.finaldata[600:800]
    pro.train5 = pro.finaldata[800:1000]
    # a1=int(0)
    # a2=int(0)
    # for i in range(len(pro.train1)):
    #     if pro.train1[i][1]==1:
    #         a1+=1
    #     else :
    #         a2+=1
    # print(a1,a2)
    # a1=int(0)
    # a2=int(0)
    # for i in range(len(pro.train2)):
    #     if pro.train2[i][1]==1:
    #         a1+=1
    #     else :
    #         a2+=1
    # print(a1,a2)
    # a1=int(0)
    # a2=int(0)
    # for i in range(len(pro.train3)):
    #     if pro.train3[i][1]==1:
    #         a1+=1
    #     else :
    #         a2+=1
    # print(a1,a2)
    # a1=int(0)
    # a2=int(0)
    # for i in range(len(pro.train4)):
    #     if pro.train4[i][1]==1:
    #         a1+=1
    #     else :
    #         a2+=1
    # print(a1,a2)
    # a1=int(0)
    # a2=int(0)
    # for i in range(len(pro.train5)):
    #     if pro.train5[i][1]==1:
    #         a1+=1
    #     else :
    #         a2+=1
    # print(a1,a2)
    Ftrain1 = pro.train5 +pro.train2+pro.train3+pro.train4
    Ftrain2 = pro.train1 +pro.train5+pro.train3+pro.train4
    Ftrain3 = pro.train1 +pro.train2+pro.train5+pro.train4
    Ftrain4 = pro.train1 +pro.train2+pro.train3+pro.train5
    Ftrain5 = pro.train1 +pro.train2+pro.train3+pro.train4
    pro.k_fold_cross(Ftrain1,pro.train1)
    pro.k_fold_cross(Ftrain2,pro.train2)
    pro.k_fold_cross(Ftrain3,pro.train3)
    pro.k_fold_cross(Ftrain4,pro.train4)
    pro.k_fold_cross(Ftrain5,pro.train5)
    print(pro.accu)
