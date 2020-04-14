import pandas as pd
import numpy as np
import math
class Process(object):
    '''
    Class implementing Naive Bayes from scratch
    It takes data as a csv input , divides the data to 5 random sets and uses
    5-fold cross validation to perform Sentiment Analysis
    '''
    def __init__(self):
        '''
        dataframe = stores data where 1 column is text, and 2 column is corresponding label
        finaldata = stores the data of dataframe after processing it, removing punctuations, stopwords etc..
        train1,2,3,4,5 = data of finaldata divided into five parts
        accu = accuracy of model
        f_score = F score of model
        recal  Recoil of model
        precision_o = Precision of model
        '''
        self.dataframe = pd.read_csv("New_data.csv")
        self.finaldata=[]
        self.train1 = []
        self.train2=[]
        self.train3=[]
        self.train4=[]
        self.train5=[]
        self.accu=[]
        self.f_score=[]
        self.recal=[]
        self.precision_o=[]

    def processTweets(self):
        '''
        processTweets converts the input data stored in dataframe ,by removing punctutations, numbers , and tokenizing it ,
        finally storing it in Final Data
        '''
        newt = []
        for i in range(len(self.dataframe)):
            self.finaldata.append([self.dataframe.iloc[i,0].lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
           .replace(";", "").replace(":", "").replace("*", "").replace("\"","")
           .replace("(", "").replace(")", "").replace("-","").replace("_","")
           .replace("/", "").replace("$","").replace("#","").replace("0","")
           .replace("2","").replace("1","").replace("3","").replace("4","")
           .replace("5","").replace("6","").replace("7","").replace("8","")
           .replace("9","").split(),int(self.dataframe.iloc[i,1])])
        '''
        Randomly shuffling final data so remove clusters of single label if present
        '''
        np.random.shuffle(self.finaldata)



    def cal_fscore(self,pred,test):
        '''
        Function calculates F score of current Fold and stores it in f_score
        tp = True Positve ( when predicted and actual both are 1 )
        tn = True Negative ( when predicted and actual both are 0)
        fp = false positive (when predicted is 1 but actual is 0)
        fn = false negative (when predicted is 0 and actual is 1)
        precision calculates Precision of this fold and stores it int precision_o
        same goes for f_score and recal
        '''

        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(pred)):
            if pred[i]==1 and test[i][1]==1:
                tp+=1
            if pred[i]==1 and test[i][1]==0:
                fp+=1
            if pred[i]==0 and test[i][1]==1:
                fn+=1
            if pred[i]==0 and test[i][1]==0:
                tn+=1
        precision=float(tp)/float(tp+fp)
        recall=float(tp)/float(tp+fn)
        fscore=float(2*precision*recall)/float(precision+recall)
        self.f_score.append(fscore)
        self.recal.append(recall)
        self.precision_o.append(precision)


    def k_fold_cross(self,train,test):
        '''
        Model that performs k-fold cross  validation,where k here is 5
        Probalitiy is calculated using two ways , bayes theorem and applyinh log to bayes theorem,
        In both cases results were similar

        '''
        words_in_cl0=0
        words_in_cl1=0
        vocab = {}
        for i in range(len(train)):
            for j in train[i][0]:
                if j in vocab:
                    continue
                else :
                    vocab[j]=1
        # print(len(vocab))
        for i in range(len(train)):
            if train[i][1]==1:
                words_in_cl1 += len(train[i][0])
            else :
                words_in_cl0 += len(train[i][0])
        dict1 = {}
        dict0 = {}
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
        a00 = words_in_cl0/len(train)
        a01 = words_in_cl1/len(train)
        pred=[]
        pred1=[]
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
        for i in range(len(test)):
            a2=a00
            a3=a01
            for j in test[i][0]:
                if j in dict0:
                    a2*=(dict0[j]/words_in_cl0)
                    cnt+=1
                else:
                    a2*=(1/(words_in_cl0+len(vocab)))
                    cnt1+=1
                if j in dict1:
                    a3*=(dict1[j]/words_in_cl1)
                    cnt+=1
                else:
                    a3*=(1/(words_in_cl1+len(vocab)))
                    cnt1+=1
            if(a2>a3):
                pred1.append(0)
            else :
                pred1.append(1)
        aa=0
        aa1=0
        # print(cnt,cnt1)
        for i in range(len(pred)):
            if int(pred[i])==test[i][1]:
                aa+=1
        for i in range(len(pred1)):
            if int(pred1[i])==test[i][1]:
                aa1+=1
        self.accu.append(aa1/200)
        self.cal_fscore(pred1,test)
        return

if __name__ == '__main__':
    pro = Process()
    pro.processTweets()
    pro.train1 = pro.finaldata[:200]
    pro.train2 = pro.finaldata[200:400]
    pro.train3 = pro.finaldata[400:600]
    pro.train4 = pro.finaldata[600:800]
    pro.train5 = pro.finaldata[800:1000]
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
    print("Accuracy : ",pro.accu,np.array(pro.accu).mean(),"+-",np.array(pro.accu).std())
    print("F-score : ",pro.f_score,np.array(pro.f_score).mean(),"+-",np.array(pro.f_score).std())
    print("Recoil : ",pro.recal,np.array(pro.recal).mean(),"+-",np.array(pro.recal).std())
    print("Precision : ",pro.precision_o,np.array(pro.precision_o).mean(),"+-",np.array(pro.precision_o).std())
