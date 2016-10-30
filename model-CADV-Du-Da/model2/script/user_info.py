# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

start=time.clock()
#####数据处理一期####
#####################
df1=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_info.csv')
df2=df1.replace({
'age':'NONE',
'sex':0,
'occupation':0,
'education':0,
'marital_status':0,
'live_info':0,
'local_hk':0,
'money_function':0,
'company_type':0,
'school_type':0,
'business_type':0,
'business_year':0,
'personnel_num':0,
'pay_type':0},np.nan)
df3=df2.groupby('user_id').agg('max').reset_index()
# df3.dropna()
df3.to_csv(r'D:\360datamining\data\firstdata\data\user_info_tf1.csv',index=False)
########处理每列数据##############
user_size=df2.groupby('user_id').size()
time_vary=df2.groupby('user_id')['tm_encode'].transform('max')-df2.groupby('user_id')['tm_encode'].transform('min')
time_mean=df2.groupby('user_id')['tm_encode'].transform('mean')
time_min=df2.groupby('user_id')['tm_encode'].transform('min')
print time_mean
##########数据处理函数#############
fill_max=lambda g: g.fillna(g.max())
fill_mode=lambda g: g.fillna(mode(g).mode[0])
############具体步骤###############
user_id=df2['user_id']
age_tf=df2.groupby('user_id')['age'].apply(fill_mode)
sex_tf=df2.groupby('user_id')['sex'].apply(fill_mode)
realtime=df2['tm_encode']
salary_tf=df2.groupby('user_id')['salary'].apply(fill_mode)
marital_status=df2.groupby('user_id')['marital_status'].apply(fill_mode)
occupation=df2.groupby('user_id')['occupation'].apply(fill_mode)
education=df2.groupby('user_id')['education'].apply(fill_mode)
live_info=df2.groupby('user_id')['live_info'].apply(fill_mode)
money_function=df2.groupby('user_id')['money_function'].apply(fill_mode)
company_type=df2.groupby('user_id')['company_type'].apply(fill_mode)
school_type=df2.groupby('user_id')['school_type'].apply(fill_mode)
local_hk=df2.groupby('user_id')['local_hk'].apply(fill_mode)
flow=df2.groupby('user_id')['flow'].apply(fill_mode)
gross_profit=df2.groupby('user_id')['gross_profit'].apply(fill_mode)
business_type=df2.groupby('user_id')['business_type'].apply(fill_mode)
business_year= df2.groupby('user_id')['business_year'].apply(fill_mode)
personnel_num= df2.groupby('user_id')['personnel_num'].apply(fill_mode)
pay_type= df2.groupby('user_id')['pay_type'].apply(fill_mode)
product_id= df2.groupby('user_id')['product_id'].apply(fill_mode)
expect_quota= df2.groupby('user_id')['expect_quota'].apply(fill_mode)
max_month_repay=df2.groupby('user_id')['max_month_repay'].apply(fill_mode)

#合并数据集
df4=pd.DataFrame({
    'user_id':user_id,
    'age':age_tf,
    'sex':sex_tf,
    'tm_encode':realtime,
    'occupation':occupation,
    'education':education,
    'marital_status':marital_status,
    'live_info':live_info,
    'local_hk':local_hk,
    'money_function':money_function,
    'company_type':company_type,
    'school_type':school_type,
    'business_type':business_type,
    'business_year':business_year,
    'personnel_num':personnel_num,
    'pay_type':pay_type,
    'flow':flow,
    'gross_profit':gross_profit,
    'salary':salary_tf,
    'expect_quota':expect_quota,
    'max_month_repay':max_month_repay,
    'product_id':product_id,


})
# df5=df4.reindex(['user_id','age','sex','max_month_repay','occupation','education','marital_status','live_info','local_hk','money_function','company_type','salary','school_type','flow','gross_profit','business_type','business_year','personnel_num','pay_type','product_id','tm_encode'])
df4.to_csv(r'D:\360datamining\data\firstdata\data\userpic_test.csv',index=False)
# df4=pd.read_csv(r'D:\360datamining\data\firstdata\data\userpic_test.csv')
df5=df4.groupby('user_id').agg('max').reset_index()


df5['size']=pd.Series(tuple(user_size))
size=np.array(tuple(user_size))
ep_qta=np.array(df5['expect_quota'])
df5['my_sum']=pd.Series(ep_qta*size)
print(df5['my_sum'])
df5['time_vary']=pd.Series(time_vary)
df5['time_mean']=pd.Series(time_mean)
df5['tm_min']=pd.Series(time_min)
print df5.head(10)

#############标准化#################

df5.to_csv(r'D:\360datamining\data\firstdata\data\user_info_tf2.csv',index=False)
table1= df1.select_dtypes(include=['O','float64','int64']).describe().T\
    .assign(missing_pct=df1.apply(lambda x : float(len(x)-x.count())/len(x)))
table1.to_csv(r'D:\360datamining\data\firstdata\data\table1.csv')
# print type(np.nan)_
table2= df5.select_dtypes(include=['O','float64','int64']).describe().T\
    .assign(missing_pct=df5.apply(lambda x : float(len(x)-x.count())/len(x)))
table2.to_csv(r'D:\360datamining\data\firstdata\data\table2.csv')
demo_data = df5.loc[:,df5.columns.str.contains("[a-zA-z]+")]
plt.figure(2)
plt.figure(1)
plt.subplot(211)
demo_data.boxplot(column=['expect_quota','salary','flow','gross_profit',],return_type='axes')
plt.subplot(212)
demo_data.boxplot(column=['sex','occupation','education','marital_status','live_info','local_hk','money_function','company_type','school_type','business_type','business_year','personnel_num','pay_type','time_vary'],return_type='axes')
plt.show()
end=time.clock()
print('Running time: %s Seconds'%(end-start))
