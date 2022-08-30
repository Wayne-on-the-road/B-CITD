# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:16:11 2022

@author: Wayne
"""

import pandas as pd
import os
import datetime 
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm,trange

#oss=ties[ties['user_id']=='CEL0561']['supervisor'][0]

def user_index(index,user):
    L=[]
    if index is not None:
        for i in range(dfn.shape[0]):
            if (dfn[i][2]==user)&(dfn[i][0]!=index):
                L.append(i)
    else:
        for i in range(dfn.shape[0]):
            if dfn[i][2]==user:
                L.append(i)
    return L

def generate_edges(index,user,relation):
    L_boss=[]
    L_shareboss=[]
    L_self= user_index(index,user)
    temp=relation[relation['user_id']==user]['supervisor'].to_list()
    boss=temp[0]
    share_boss= relation[relation['supervisor']==boss]['user_id'].to_list()
    temp= relation[relation['employee_name']==boss]['user_id'].to_list()
    if len(temp)>0:
        boss_id=temp[0]
        L_boss= user_index(None,boss_id)
        for co_worker in share_boss:
            if co_worker != user:
                L_shareboss=L_shareboss+user_index(None,co_worker) 
    L_total= L_self+L_boss+L_shareboss
    #if len(user_index(None,boss_id)) >0:
    return L_total

def df_split(df,rate):
    df_train= df.sample(frac=rate)
    df_test= pd.concat([df,df_train],ignore_index= True).drop_duplicates(keep=False)
    return df_train,df_test

path=r'D:\Users\Wayne\PyProgramming\cert-try\github\CERT4.2'
filepath=os.path.join(path,'userday-test.csv')    

#filepath=r'D:\CERT_database\r4.2\user_feature_label\1-data-test-rebiuld1.csv'
df= pd.read_csv(filepath)
df1= df[df['label']==1]
size1=len(df1)
df0= df[df['label']==0]
#size0= len(df0)

rate_bad=0.5 # percent for mixing:0.5 are insiders
rate_train=0.7 # percent for training
df0= df0.sample(int(size1/rate_bad)-size1)

df1_train,df1_test = df_split(df1, rate_train)
df0_train,df0_test = df_split(df0, rate_train)

# df_train1=df1.random_split(rate_train,1-rate_train)
# df_train0=df0.sample(int(rate_train*size1/rate_bad)) 
# df_test1=df1.sample(int((1-rate_train)*size1))
# df_test0=df0.sample(int((1-rate_train)*size1/rate_bad)) 

df_train=pd.concat([df1_train,df0_train],ignore_index=True)
df_train.sort_values(by='index',inplace= True, ascending=True)
#print(df_train)
df_test=pd.concat([df1_test,df0_test],ignore_index=True)
df_test.sort_values(by='index',inplace= True, ascending=True)
#print(df_test)
df_train.to_csv(filepath.replace('userday-test', 'data-train'),index=False)
df_test.to_csv(filepath.replace('userday-test', 'data-test'),index=False)

# df_train=pd.read_csv(filepath.replace('rebiuld1', 'train'))
# df_test=pd.read_csv(path.replace('rebiuld1', 'test'))

df_total=pd.concat([df_train,df_test],ignore_index=True)
df_total.sort_values(by='index',inplace= True, ascending=True)
df_total.to_csv(filepath.replace('userday-test', 'data-total'),index=False)


# data= pd.read_csv(os.filepath.join(new_filepath, '1-data.csv'))
# df_total= pd.read_csv(filepath.replace('rebiuld1', 'combine'))
# print(df_total)

relation=pd.read_csv(os.path.join(path,'userlist-test.csv'))

dfn=df_total[['index','date_index','user_index']].to_numpy()

edge= pd.DataFrame(columns=['start','end'])
start=[]
end=[]
for i in trange(dfn.shape[0]):
    L_end= generate_edges(dfn[i][0],dfn[i][2],relation)
    L_start= [i]*len(L_end)
    start= start + L_start
    end= end + L_end

edge['start']=start
edge['end']=end
#edge.to_csv(filepath.replace('userday-test', 'edge_index'),index=False)

#convert directed edge to undirected:
#directed=pd.read_csv(filepath.replace('rebiuld1', 'edge_index'))
temp= pd.DataFrame(columns=['start','end'])
temp['start']= edge['end']
temp['end']= edge['start']

undirected= pd.concat([edge,temp],ignore_index=True)
undirected.to_csv(filepath.replace('userday-test', 'undirected_edge'),index=False)


# print(len(set(directed['start'])))

    #edge_index.append(temp)
    #out= np.array(edge_index)
    #print(out.shape)
    

# for index,row in df_total.iterrows():
    

# for node in tqdm(nodes):
#     if not (column.endswith('f5') or column.endswith('label')):
#         data[column]= data[column].apply(convert_cell)
            
# data.to_csv(os.filepath.join(new_filepath,'1-data-test.csv'))



