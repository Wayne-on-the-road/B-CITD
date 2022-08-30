# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:18:41 2022

@author: Wayne
"""


import os
from sklearn import metrics,svm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
#from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,auc,roc_curve



def extract_data(data):
    col_list= data.columns.values.tolist()
    label_col= [i for i in range(len(col_list)) if col_list[i]=='label'][0]
    data_np= data.to_numpy()
    _x=[]
    _y=[]
    
    for i in range(len(data_np)):
        _y.append(data_np[i][label_col])
        for j in range(len(col_list)):
            temp= data_np[i][label_col+1:].tolist()
        _x.append(temp)
    return _x,_y

# class train_dataset(Data.Dataset):
#     def __init__(self,raw_data):
#         self.col_list= raw_data.columns.values.tolist()
#         self.label_col= [i for i in range(len(self.col_list)) if self.col_list[i]=='label'][0]
#         self.raw_np= raw_data.to_numpy()                      
    
        
    
    
#     def __getitem__(self, idx):
#         label= self.raw_np[idx][self.label_col]
#         feature= self.raw_np[idx][self.label_col+1:].tolist()
#         feature= torch.unsqueeze(torch.tensor(feature),dim=0)
#         return feature,torch.tensor(label)
        
#     def __len__(self):
#         return len(self.raw_np)
path=r'D:\Users\Wayne\PyProgramming\cert-try\github\CERT4.2'

#load data from without graph feature to train:
  
train_data= pd.read_csv(os.path.join(path,'data-train.csv'))
test_data= pd.read_csv(os.path.join(path,'data-test.csv'))

train_x, train_y = extract_data(train_data)

test_x,test_y= extract_data(test_data)


   # use default parameters:
svc = svm.SVC(probability=(True))

svc.fit(train_x,train_y)
pred_y = svc.predict(test_x)
pos_score = svc.predict_proba(test_x)[:,1]
true_y = test_y

accuracy = accuracy_score(true_y, pred_y)    
precision = precision_score(true_y, pred_y)
recall = recall_score(true_y, pred_y)
f1 = f1_score(true_y, pred_y)


#try graph feature data
train_data= pd.read_csv(os.path.join(path,'data-train-g.csv'))
test_data= pd.read_csv(os.path.join(path,'data-test-g.csv'))

train_x, train_y = extract_data(train_data)

test_x,test_y= extract_data(test_data)

svc = svm.SVC(probability=(True))

svc.fit(train_x,train_y)
pred_y_g = svc.predict(test_x)
pos_score_g = svc.predict_proba(test_x)[:,1]
true_y_g = test_y

accuracy_g = accuracy_score(true_y_g, pred_y_g)    
precision_g = precision_score(true_y_g, pred_y_g)
recall_g = recall_score(true_y_g, pred_y_g)
f1_g = f1_score(true_y_g, pred_y_g)




fpr, tpr, threshold = roc_curve(true_y,pos_score)
roc_auc = auc(fpr,tpr)  

fpr_g, tpr_g, threshold_g = roc_curve(true_y_g,pos_score_g)
roc_auc_g = auc(fpr_g,tpr_g)   

#print('roc_auc:', roc_auc)
lw = 1.5
#plt.subplot(5,3,2)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='SVM (AUC = %0.4f)' % roc_auc)  
plt.plot(fpr_g, tpr_g, color='red',
         lw=lw, label='SVM + Graph Feature (AUC = %0.4f)' % roc_auc_g)  

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random', x=0.6,y=0.4)
plt.legend(loc="lower right")

plt.show()

print(accuracy,precision,recall ,f1)

print(accuracy_g,precision_g ,recall_g ,f1_g)



print('finished training!')


    
