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

# generate datetime type list for use: 
date= pd.date_range(start= '2010-01-01', end= '2011-05-20')
date= date.tolist()
date= pd.DataFrame({'date':date})
date['date']= pd.to_datetime(date['date'],format='%Y-%m-%d %H:%M:%S')
date_list= date.iloc[:,0].to_list()


# read in file and conver the cells containning time to datetime type:
    
# #df= pd.read_csv(r'D:\CERT_database\r4.2\logon.csv',nrows=(10000))
# df= pd.read_csv(r'D:\CERT_database\r4.2\logon.csv')
# df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
# #device= pd.read_csv(r'D:\CERT_database\r4.2\device.csv',nrows=(10000))
# device= pd.read_csv(r'D:\CERT_database\r4.2\device.csv')
# device['date']= pd.to_datetime(device['date'],format='%m/%d/%Y %H:%M:%S')


# combine all user file into one(need to give the directory for all files):
def combine(filepath):
    files= os.listdir(filepath)
    data=pd.DataFrame()
    i= 0
    for file in tqdm(files):
        csv_path= os.path.join(filepath,file)
        d_csv= pd.read_csv(csv_path)
        if i<1:
            data= pd.concat([data,d_csv],axis=1,join='outer')
            i= i+1
            continue
        d_csv=d_csv.drop(labels= 'day',axis=1)
        data= pd.concat([data,d_csv],axis=1,join='outer')
    return data
    #data.to_csv(os.path.join(filepath,'userday.csv'),index=False)


#Wayne's note:convert each cell value to datatime and then map it to 0-1,input is dataframe cell:
def convert_cell(cell):
    cell= pd.to_datetime(cell,format='%Y/%m/%d %H:%M:%S')
    cell= (cell.hour*60 + cell.minute)/(24*60)
    return cell

# convert dataframe's 'date' column to datetime tpye,return a dateframe:
def convert_time(path):
    df= pd.read_csv(path)
    df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
    return df

# drop unwanted rows and columns, fill blank with 0:
def data_fill_drop(data):
    #data= pd.read_csv(filepath)
    data= data.fillna(0)
    data= data.drop(labels='Unnamed 0:',axis=1)
    data= data.drop(index= [0,357,358,359,360])
    return data

# add label for each user file, based on "day-user" sample
# (u_path:location for user files,insider_path:location for label files,
#  u_list:user list for processing, insider_df:load of csv file summary for all insiders
# date_list: date_list for common use without any filtering
def add_label(u_path,insider_path,u_list,insider_df,date_list):
    for user in (u_list):
        label=[0]*len(date_list)
        df= insider_df[insider_df['user']== user]
        if not df.empty:
            filepath= os.path.join(insider_path,df.iloc[0]['details'])
            df= pd.read_csv(filepath,names= ['file','id','date','user','pc','activity'],usecols=([0,1,2,3,4,5]),header=None)
            df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
            for day in df['date']:
                for i in range(len(date_list)):
                    if date_list[i] >= day:
                        label[i-1]= 1
                        break
            
        df_out= pd.DataFrame({'label':label})
        df_out['date']=date_list
        df_out['week']= df_out['date'].dt.dayofweek
        df_out= df_out[df_out['week'].isin([0,1,2,3,4])] 
        df_out= df_out.drop(labels=['week'],axis=1)
        in_path= os.path.join(u_path,user + '.csv')
        df_in=pd.read_csv(in_path)
        df_in.insert(1,user + '-label', df_out['label'].to_list())
        df_in.to_csv(in_path.replace('user_feature', 'user_feature_labeled'),index=False)
        #print(df_in)
        #print(df_out)

#funtion for re-organizing data from day-based to day&user based:
def re_build(df,head):
    size= len(head)-3
    col_list= df.columns.values.tolist()
    df= df.to_numpy()
    L_new=[]
    # ID=[]
    for i in range(df.shape[0]):
        for j in range(1,df.shape[1],size):
            temp=np.hstack([np.array(str(i)),np.array(col_list[j].strip('-label')),df[i][0],df[i][j:j+size]])
            #ID.append(str(i)+'-'+col_list[j].strip('-label'))
            L_new.append(temp)
    df_new= pd.DataFrame(L_new,columns=head)
    #print(df_new)
    return df_new

       
# filepath= r'D:\CERT_database\r4.2\user_feature\1-data-test.csv'    
# data= data_fill_drop(filepath)
# print(data)
# data.to_csv(filepath.replace('.csv', '-new.csv'),index=False)


if __name__ == '__main__':
    #mp.freeze_support()
    path=r'D:\Users\Wayne\PyProgramming\cert-try\github\CERT4.2'
    filepath=os.path.join(path,'user_feature')
    insider_path= os.path.join(path,'answer')
    insider_df= pd.read_csv(os.path.join(insider_path,'insiders.csv'))
    files= os.listdir(filepath)
    u_list= [file.strip('.csv') for file in files]
    num_u= len(u_list)
    num_p= int(mp.cpu_count()/2)
    block= int(num_u/num_p)
    #block= 1
    L_block= []
    for i in range(0,num_u,block):
        piece= u_list[i:i+block]
        L_block.append(piece)
    
    pool= mp.Pool(num_p)
    start= time.time()
    for b in L_block:
        pool.apply_async(add_label,args=(filepath, insider_path,u_list,insider_df,date_list))
    pool.close()
    pool.join()
    end= time.time()
    print('add label time:',end-start)
    
    #add_label(filepath,insider_path,u_list,insider_df,date_list)
    
    filepath_labeled= os.path.join(path,'user_feature_labeled')
    
    data = combine(filepath_labeled)
    data.to_csv(os.path.join(path,'daybased.csv'),index=False)
#Wayne's note:following part is about how to convert time to number:
   
    # data= pd.read_csv(os.path.join(new_path, '1-data.csv'))
    # for column in tqdm(data.columns[1:]):
    #     if not (column.endswith('f5') or column.endswith('label')):
    #         data[column]= data[column].apply(convert_cell)
            
    # data.to_csv(os.path.join(new_path,'1-data-test.csv'))
 
#Wayne's note:following part is about how to fill and drop data(same as function):
    file= os.path.join(path,'daybased.csv')
    data= pd.read_csv(file)
    data= data.fillna(0)
    #data= data.drop(labels=['Unnamed: 0'],axis=1)
    data= data.drop(index= [0,357,358,359,360])
    #print(data)
    data.to_csv(file.replace('.csv', '-new.csv'),index=False)

#Wayne's note: rebiuld final data to new formate with date_index, user_index,and date and lable:
    df_re_biuld= pd.read_csv(os.path.join(path, 'daybased-new.csv'))
    head=['date_index','user_index','date','label','f1','f2','f3','f4','f5']
    df_re_biuld= re_build(df_re_biuld,head)
    df_re_biuld.to_csv(os.path.join(path,'userday.csv'),index_label='index')
    

