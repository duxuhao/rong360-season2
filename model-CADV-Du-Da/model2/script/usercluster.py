# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans       #导入K-means算法包
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
####################用户基本属性/合并表/统计0##################
user=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_info_tf2.csv')
# del user['Unnamed: 0'],#user['Unnamed: 0.1']

print user.head(5)
usernormal= user.select_dtypes(include=['O','float64','int64']).describe().T\
    .assign(missing_pct=user.apply(lambda x : float(x[x==0].count())/len(x)))
usernormal.to_csv(r'D:\360datamining\data\firstdata\data\usernormal.csv')

###################用户行为和统计#################

user_action1=pd.read_csv(r'D:\360datamining\data\firstdata\data\consumption_recode.csv')
user_action2=user_action1.groupby('user_id')
user_action_missing_stats=user_action1.select_dtypes(include=['O','float64','int64']).describe().T\
    .assign(missing_pct=user_action1.apply(lambda x : float(x[x==0].count())/len(x)))
user_action_missing_stats.to_excel(r'D:\360datamining\data\firstdata\data\result\actioncount0.xlsx')



###################统计基本属性0个数###############
table1= user.select_dtypes(include=['O','float64','int64']).describe().T\
    .assign(missing_pct=user.apply(lambda x : float(x[x==0].count())/len(x)))
table1.to_excel(r'D:\360datamining\data\firstdata\data\result\usercount0.xlsx')

# ###################合并社交网络##################
# net=pd.read_csv(r'D:\360datamining\data\firstdata\data\relation2.csv')
# userlist.to_csv(r'D:\360datamining\data\firstdata\data\result\userlist.csv',index=False)
# socialnet=pd.merge(net,userlist,on='user_id',how='inner')
# socialnet1=socialnet[socialnet['relation2_type']==1]
# socialnet2=socialnet[socialnet['relation2_type']==2]
# socialnet3=socialnet[socialnet['relation2_type']==3]
# del socialnet1['relation2_type'],socialnet1['time'],socialnet2['relation2_type'],socialnet2['time'],socialnet3['relation2_type'],socialnet3['time']
# socialnet1.rename(columns={'user2_id':'Source','user_id':'Target','relation2_weight':'Weight'},inplace=True)
# socialnet2.rename(columns={'user2_id':'Source','user_id':'Target','relation2_weight':'Weight'},inplace=True)
# socialnet3.rename(columns={'user2_id':'Source','user_id':'Target','relation2_weight':'Weight'},inplace=True)
# socialnet1.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type1.csv',index=False)
# socialnet2.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type2.csv',index=False)
# socialnet3.to_csv(r'D:\360datamining\data\firstdata\data\relationship2\type3.csv',index=False)
######################用户聚类########################
data=user.loc[:,['tm_encode','salary','age','occupation', 'marital_status','education','live_info','local_hk','expect_quota','size']]
# traindata=data[(data['expect_quota']<1866513)&(data['salary']<268447566)&(data['expect_quota']>0)&(data['salary']>0)&(data['age']>12)&(data['age']<44)]
# data2=normalize(data)
maxmin=StandardScaler().fit(data)
data_X=maxmin.transform(data)

data=pd.DataFrame(data_X,index=data.index)
data2=data.apply(lambda x:(x-x.mean())**2)
data3=data2.apply(lambda x:sp.sqrt(x.sum()),axis=1)
print data3.head(5),len(data3)
eps=data3.mean()
print eps
plt.plot(range(0,len(data3)),data3)
plt.show()
# inertia=[]
# for k in range(2,20):
#     num_cluster=k
#     km=KMeans(n_clusters=k,init='k-means++',n_init=10)
#     km.fit(data)
#     print set(km.labels_)
#     inertia.append(km.inertia_)
#     print k,km.labels_
#     print km.cluster_centers_,km.inertia_
# Compute DBSCAN
db = DBSCAN(eps=1.2, min_samples=10).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print labels
print n_clusters_
user['type']=pd.Series(labels,index=user.index)
print user.head(5)
######################用户聚类添加到user##############################
user.to_csv(r'D:\360datamining\data\firstdata\data\user_info_tf2.csv')
print 'new finish'
label=pd.read_csv(r'D:\360datamining\data\firstdata\data\train.csv')
cluster=pd.merge(user,label,on='user_id',how='inner')

userlist=pd.DataFrame({'user_id':user['user_id']})

cluster.to_csv(r'D:\360datamining\data\firstdata\data\result\cluster.csv',index=False)
##################根据产品划分用户#################
# user_pdt_id_1=cluster[cluster['product_id']==1]
# user_pdt_id_2=cluster[cluster['product_id']==2]
# plt.figure(2)
# plt.plot(range(2,20),inertia, 'r', linewidth=2)
# plt.show()
# plt.figure(1)
# plt.subplot(231)
# plt.title(u"产品1用户年龄统计",fontproperties='SimHei')
# plt.hist(user_pdt_id_1['age'])
# plt.xticks((1,2))
# plt.subplot(232)
# plt.title(u"产品1期望",fontproperties='SimHei')
# plt.hist(user_pdt_id_1['expect_quota'])
# plt.subplot(233)
# plt.title(u"产品1学历",fontproperties='SimHei')
# plt.hist(user_pdt_id_1['education'])
# plt.subplot(234)
# plt.title(u"产品2用户年龄统计",fontproperties='SimHei')
# plt.hist(user_pdt_id_2['age'])
# plt.subplot(235)
# plt.title(u"产品2性别",fontproperties='SimHei')
# plt.hist(user_pdt_id_2['sex'])
# plt.subplot(236)
# plt.title(u"产品2学历",fontproperties='SimHei')
# plt.hist(user_pdt_id_2['education'])
# plt.show()