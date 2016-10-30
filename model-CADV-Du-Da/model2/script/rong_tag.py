# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
# user_action=pd.read_csv(r'D:\360datamining\data\firstdata\data\consumption_recode.csv')
user=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_add_rls1.csv')

print (u"结束")

rong_tag=pd.read_csv(r'D:\360datamining\data\firstdata\data\rong_tag.csv')
user_list=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\userlist.csv')
rong_tag_temp=pd.DataFrame(columns=['user_id','tag_max','tag_min','tag_median','tag_size'])
rong_tag_temp.user_id=rong_tag.groupby('user_id').count().index
rong_tag_temp.tag_max=rong_tag.groupby('user_id').max().values
rong_tag_temp.tag_min=rong_tag.groupby('user_id').min().values
rong_tag_temp.tag_median=rong_tag.groupby('user_id').median().values
rong_tag_temp.tag_size=rong_tag.groupby('user_id').count().values

print len(rong_tag_temp)
out=pd.merge(user,rong_tag_temp,on='user_id',how='left')
out.fillna(out.median(),inplace=True)
out.to_csv(r'D:\360datamining\data\firstdata\data\user_add_tag.csv')
#####s输出
print out.head(5)
print len(out)
out1=out.loc[:,['tm_encode','salary','age','live_info','local_hk','expect_quota','size','relation1_num','time_vary','tag_max','my_sum']]
# maxmin=MinMaxScaler().fit(out1)
# data_X=maxmin.transform(out1)
# data_X=pd.DataFrame(data_X,columns=['tm_encode','salary','age','live_info','local_hk','expect_quota','size','relation1_num','time_vary','tag_max','my_sum'])
pd.scatter_matrix(out1,diagonal='kde',color='k',alpha=0.3)
plt.savefig(r'D:\360datamining\data\firstdata\data\result\data_X.pdf',dpi=1000)
plt.show()
#
# user_action=pd.merge(user_action,user_list,on='user_id',how='inner')#重要特征/但存在缺失
# user_action_size=user_action.groupby('user_id').size()
# plt.hist(rong_tag_4race_group['rong_tag'])
# plt.show()

