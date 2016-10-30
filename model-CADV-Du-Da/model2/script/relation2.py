# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
import matplotlib.pyplot as plt
def datasplit(filename):
    df=pd.read_csv(filename)
    df1=df[df.relation2_type == 1]
    df2=df[df.relation2_type == 2]
    df3=df[df.relation2_type == 3]
    del df1['relation2_type'],df2['relation2_type'],df3['relation2_type'],
    df1.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type1.csv',index=False,encoding='utf-8')
    df2.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type2.csv',index=False,encoding='utf-8')
    df3.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type3.csv',index=False,encoding='utf-8')
    df1=pd.read_csv(r'D:\360datamining\data\firstdata\data\relationship2\type1.csv',encoding='utf-8')
    return df1,df2,df3
def dtconcat(df,list):
    relationtemp=pd.DataFrame(columns=['user_id','relation2_num','relation2_tm'])
    relationtemp.user_id=df.groupby('user2_id').count().index
    relationtemp.relation2_num=df.groupby('user2_id').count().values
    relationtemp.relation2_tm=df.groupby('user2_id')['time'].median().values
    relation=pd.merge(relationtemp,list,on='user_id',how='inner')
    print len(relation)
    print relation.head(5)
    return relation
def main():
    df1,df2,df3=datasplit(r'D:\360datamining\data\firstdata\data\relation2.csv')
    relation2=pd.read_csv(r'D:\360datamining\data\firstdata\data\relation2.csv')
    out=relation2.groupby('user2_id').size()
    print len(out)
    print 'first step finish'
    user_list=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\userlist.csv')
    relation2_type1=dtconcat(df1,user_list)
    relation2_type2=dtconcat(df2,user_list)
    relation2_type3=dtconcat(df3,user_list)
    # concat=pd.merge(user,relation,on='user_id',how='left')


    plt.show()


if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print('Running time: %s Seconds'%(end-start))


