# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:58:54 2022

@author: Wayne
"""

import torch
import os
import torch.nn as nn
#import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
#from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,roc_curve,accuracy_score,auc

EPOCH = 301
BATCH_SIZE= 50
LR= 0.01
step=2
r=0.7

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


# def extract_testdata(test_data):
#     col_list= raw_data.columns.values.tolist()
#     label_col= [i for i in range(len(col_list)) if col_list[i]=='label'][0]
#     raw_np= raw_data.to_numpy()
#     test_x=[]
#     test_y=[]
    
#     for i in range(len(raw_np)):
#         test_y.append(raw_np[i][label_col])
#         for j in range(len(col_list)):
#             temp= raw_np[i][label_col+1:].tolist()
#         test_x.append(temp)
#     return test_x,test_y

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

train_x = torch.unsqueeze(torch.tensor(train_x),dim=1)  
train_y = torch.tensor(train_y)

test_x = torch.unsqueeze(torch.tensor(test_x),dim=1)  
test_y = torch.tensor(test_y)




class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                              
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      
        )
        self.output = nn.Linear(32,2)
        

    def forward(self, x):
        out = self.conv1(x)                  
        out = self.conv2(out)                
        out = out.view(out.size(0),-1)       
        out = self.output(out)
        return F.softmax(out,dim=1)


cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR,)
loss_func = nn.CrossEntropyLoss()
L_pos=np.empty([0,572])
L_acc=[]
L_pre=[]
L_rec=[]
L_f1=[]

for epoch in range(EPOCH):
    output = cnn(train_x)
    loss = loss_func(output,train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch % step ==0) and (epoch > r*EPOCH):
        cnn.eval()
        test_output = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()# wayne' note: return max one's index in dimension 1
        true_y =test_y.tolist()
        l_out= np.array(test_output[:,1].tolist())
        l_out= l_out[np.newaxis]
        L_pos= np.append(L_pos,l_out,axis=0)
        L_acc =np.append(L_acc,accuracy_score(true_y, pred_y)) 
        L_pre=np.append(L_pre,precision_score(true_y, pred_y))
        L_rec=np.append(L_rec,precision_score(true_y, pred_y))
        L_f1=np.append(L_f1,f1_score(true_y, pred_y))
        
true_y= test_y.tolist()                      
pos_score= np.mean(L_pos,axis=0)
accuracy= np.mean(L_acc)
precision= np.mean(L_pre)
recall= np.mean(L_rec)
f1= np.mean(L_f1)
print('finished CNN with out graph feature!')

# load data with graph features to compare:

train_data= pd.read_csv(os.path.join(path,'data-train-g.csv'))
test_data= pd.read_csv(os.path.join(path,'data-test-g.csv'))

train_x, train_y = extract_data(train_data)


test_x,test_y= extract_data(test_data)

train_x = torch.unsqueeze(torch.tensor(train_x),dim=1)  
train_y = torch.tensor(train_y)

test_x = torch.unsqueeze(torch.tensor(test_x),dim=1)  
test_y = torch.tensor(test_y)

cnn_g = CNN()

optimizer = torch.optim.Adam(cnn_g.parameters(),lr=LR,)
loss_func = nn.CrossEntropyLoss()
L_pos=np.empty([0,572])
L_acc=[]
L_pre=[]
L_rec=[]
L_f1=[]

for epoch in range(EPOCH):
    
    output = cnn_g(train_x)
    loss = loss_func(output,train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch % step ==0)and (epoch > r*EPOCH):
        cnn_g.eval()
        test_output = cnn_g(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()# wayne' note: return max one's index in dimension 1
        true_y =test_y.tolist()
        l_out= np.array(test_output[:,1].tolist())
        l_out= l_out[np.newaxis]
        L_pos= np.append(L_pos,l_out,axis=0)
        L_acc =np.append(L_acc,accuracy_score(true_y, pred_y)) 
        L_pre=np.append(L_pre,precision_score(true_y, pred_y))
        L_rec=np.append(L_rec,precision_score(true_y, pred_y))
        L_f1=np.append(L_f1,f1_score(true_y, pred_y))
true_y_g= test_y.tolist()                       
pos_score_g= np.mean(L_pos,axis=0)
accuracy_g= np.mean(L_acc)
precision_g= np.mean(L_pre)
recall_g= np.mean(L_rec)
f1_g= np.mean(L_f1)



fpr, tpr, threshold = roc_curve(true_y,pos_score)
roc_auc = auc(fpr,tpr)   

fpr_g, tpr_g, threshold_g = roc_curve(true_y_g,pos_score_g)
roc_auc_g = auc(fpr_g,tpr_g)  

#print('roc_auc:', roc_auc)
lw = 1.5
plt.plot(fpr, tpr, color='grey',
         lw=lw, label='CNN (AUC = %0.4f)' % roc_auc) 
plt.plot(fpr_g, tpr_g, color='red',
         lw=lw, label='CNN + Graph Feature (AUC = %0.4f)' % roc_auc_g)  

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

print('finish training')