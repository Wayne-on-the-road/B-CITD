# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:15:31 2022

@author: Wayne
"""
import pandas as pd
import os
import datetime 
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm,trange

def convert_cell(cell):
    #cell= pd.to_datetime(cell,format='%Y/%m/%d %H:%M:%S')
    cell= (cell.hour*60 + cell.minute)/(24*60)
    return cell

def convert_time(path):
    df= pd.read_csv(path)
    df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
    return df

#create first a csv file for each user as a node,feature1 is:first logon time:
def create_node_feature(filepath,n_node,u_list,date_list,df):
    i=0
    t_delta= datetime.timedelta(1)
    for user in tqdm(u_list):
        if i< n_node:
            out1= pd.DataFrame()
            out2= pd.DataFrame()
            for date in date_list:
                #extract feature1:first Logon time
                df1= df[(date <= df['date']) & (df['date']< date + t_delta) & (df['user']== user)]
                df2= df1[df1['activity']=='Logon']
                day_out={'day':date,'feature':np.nan}
                if not df2.empty:
                    day_out['feature']= convert_cell(df2.iloc[0]['date'])
                out1= out1.append(day_out,ignore_index=True)
                #extract feature2:last Logoff time
                df2= df1[df1['activity']=='Logoff']
                day_out= {'day':date,'feature':np.nan}
                #print(df1)
                if not df2.empty:
                    day_out['feature']= convert_cell(df2.iloc[-1]['date'])
                out2= out2.append(day_out,ignore_index=True)
            out1= out1.rename(columns={'feature':user+'-f1'})
            out2= out2.rename(columns={'feature':user+'-f2'})
            out2= out2.drop(labels=('day'),axis=1)
            out= pd.concat([out1,out2],axis=1,join='outer')
            out.to_csv(os.path.join(filepath, user + '.csv'),index=False)
            #print(out)
            i=i+1          

def add_feature_device(filepath,u_list,date_list,df):
    d_delta= datetime.timedelta(days=1)
    h_delta= datetime.timedelta(hours=1)
    #files= os.listdir(filepath)
    for user in tqdm(u_list):
        #user= u.strip('.csv')
        out3= pd.DataFrame()
        out4= pd.DataFrame()
        out5= pd.DataFrame()
        for date in date_list:
            #extract first activity time:
            df1= df[(date <= df['date']) & (df['date']< date + d_delta) & (df['user']== user)]
            day_out= {'day':date,'feature':np.nan}
            if not df1.empty:
                day_out['feature']= convert_cell(df1.iloc[0]['date'])
            out3= out3.append(day_out,ignore_index=True)
            #extract last activity time:
            day_out= {'day':date,'feature':np.nan}
            if not df1.empty:
                day_out['feature']= convert_cell(df1.iloc[-1]['date'])
            out4= out4.append(day_out,ignore_index=True)
            #extract offhour activity times:
            day_out= {'day':date,'feature':np.nan}
            if not df1.empty:
                df2=df1[(df1['date']< date + 8*h_delta) | (date + 18*h_delta < df1['date'])]
                if not df2.empty:
                    day_out['feature']=len(df2)
                # count= dict(df2['activity'].value_counts())
                # day_out['feature']=count['Connect']
            out5= out5.append(day_out,ignore_index=True)
        out3= out3.rename(columns={'feature':user+'-f3'})
        out4= out4.rename(columns={'feature':user+'-f4'})
        out5= out5.rename(columns={'feature':user+'-f5'})
        out3= out3.drop(labels=('day'),axis=1)
        out4= out4.drop(labels=('day'),axis=1)
        out5= out5.drop(labels=('day'),axis=1)
        csv_path= os.path.join(filepath,user+'.csv')
        d_csv= pd.read_csv(csv_path)
        d_csv= pd.concat([d_csv,out3,out4,out5],axis=1,join='outer')
        #print(d_csv)
        d_csv.to_csv(csv_path,index=False)
        #print(out)

        
if __name__ == '__main__':
    #mp.freeze_support()
    #generate date_list for extracting feature:only weekday
    date= pd.date_range(start= '2010-01-01', end= '2011-05-20')
    date= date.tolist()
    date= pd.DataFrame({'date':date})
    date['date']= pd.to_datetime(date['date'],format='%Y-%m-%d %H:%M:%S')
    date['week']= date['date'].dt.dayofweek
    date= date[date['week'].isin([0,1,2,3,4])] 
    date= date.drop(labels='week',axis=1)
    date_list= date.iloc[:,0].to_list()

    filepath=r'D:\Users\Wayne\PyProgramming\cert-try\github\CERT4.2'

    #generate user list for extracting feature:multiprocessing based
    
    u= pd.read_csv(os.path.join(filepath,'userlist.csv'))
    u_list= u['user_id'].to_list()
    num_u= len(u_list)
    num_p= int(mp.cpu_count()/2)
    block= int(num_u/num_p)
    #block= 1
    L_block= []
    for i in range(0,num_u,block):
        piece= u_list[i:i+block]
        L_block.append(piece)
    #print(L_block[-1])    
    
    pool= mp.Pool(num_p)
    start0= time.time()
    paths=[os.path.join(filepath,'logon.csv'),os.path.join(filepath,'device.csv')]
    #paths=[r'D:\CERT_database\r4.2\logon.csv']    
    results= [pool.apply_async(convert_time,args=(path,)) for path in paths]                      
    pool.close()
    pool.join()
    logon= results[0].get()
    device= results[1].get()
    end0= time.time()
    print('convert_time:',end0-start0)

    #create node with multiprocessing method:
    pool= mp.Pool(num_p)
    start= time.time()
    for b in L_block:
        pool.apply_async(create_node_feature,args=(os.path.join(filepath,'user_feature'),block, b, date_list, logon))
    pool.close()
    pool.join()
    end= time.time()
    print('create_node time:',end-start)


    #add new feature with multiprocessing method:
    files= os.listdir(os.path.join(filepath,'user_feature'))
    u_list1= [file.strip('.csv') for file in files]    
    
    pool= mp.Pool(num_p)
    start= time.time()
    for b in L_block:
        pool.apply_async(add_feature_device,args=(os.path.join(filepath,'user_feature'), b, date_list, device))
    pool.close()
    pool.join()
    end= time.time()
    print('add_feature_device time:',end-start)

