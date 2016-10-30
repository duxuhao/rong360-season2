# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
from scipy.stats import mode
import matplotlib.pyplot as plt
import math
usercost1=pd.read_csv(r'D:\360datamining\data\firstdata\data\consumption_recode.csv')
user=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_aft_datawash.csv')
userrelation2=pd.read_csv(r'D:\360datamining\data\firstdata\data\relation2.csv')
######消费数据###############
cost=pd.DataFrame(columns=['user_id','cost_size','credit_lmt_amt','curt_jifen','current_bill_bal','current_bill_min_repay_amt','current_bill_amt','credit_lmt_amt_max','curt_jifen_max','prior_period_bill_amt','prior_period_repay_amt',])
cost.user_id=usercost1.groupby('user_id').count().index
cost.cost_size=usercost1.groupby('user_id').count().values
cost.credit_lmt_amt=usercost1.groupby('user_id')['credit_lmt_amt'].mean().values
cost.curt_jifen=usercost1.groupby('user_id')['curt_jifen'].mean().values
cost.current_bill_bal=usercost1.groupby('user_id')['current_bill_bal'].median().values
cost.current_bill_min_repay_amt=usercost1.groupby('user_id')['current_bill_min_repay_amt'].median().values
cost.current_bill_amt=usercost1.groupby('user_id')['current_bill_amt'].median().values
cost.credit_lmt_amt_max=usercost1.groupby('user_id')['credit_lmt_amt'].max().values
cost.curt_jifen_max=usercost1.groupby('user_id')['curt_jifen'].max().values
cost.prior_period_bill_amt=usercost1.groupby('user_id')['prior_period_bill_amt'].median().values
cost.prior_period_repay_amt=usercost1.groupby('user_id')['prior_period_repay_amt'].median().values
out=pd.merge(user,cost,on='user_id',how='left')

print out['cost_size'].median()
def tf(x):
    return -float(x/253006)*np.log(float(x/253006))
# t=out.credit_lmt_amt.median()
# print t{'cost_size':16.0,'credit_lmt_amt':610928.998009}
out['cost_size']=out['cost_size'].replace(np.nan,16.0)
out['credit_lmt_amt']=out['credit_lmt_amt'].replace(np.nan,610928.998009)
out['curt_jifen']=out['curt_jifen'].replace(np.nan,out.curt_jifen.median())
out['current_bill_bal']=out['current_bill_bal'].replace(np.nan,out.current_bill_bal.median())
out['current_bill_min_repay_amt']=out['current_bill_min_repay_amt'].replace(np.nan,out.current_bill_min_repay_amt.median())
out['current_bill_amt']=out['current_bill_amt'].replace(np.nan,out.current_bill_amt.median())
out['credit_lmt_amt_max']=out['credit_lmt_amt_max'].replace(np.nan,out.credit_lmt_amt_max.median())
out['curt_jifen_max']=out['curt_jifen_max'].replace(np.nan,out.curt_jifen_max.median())
out['prior_period_bill_amt']=out['prior_period_bill_amt'].replace(np.nan,out.credit_lmt_amt_max.median())
out['prior_period_repay_amt']=out['prior_period_repay_amt'].replace(np.nan,out.curt_jifen_max.median())
print out.head(5)

out.to_csv(r'D:\360datamining\data\firstdata\data\user_add_cost.csv')
##############社交数据2###################

#
# def  logtu(name,df):
#     df1=df[df[name]>=0]
#     df2=df[df[name]<0]
#     df2[name]=-df2[name]
#     df2[name]=df2[name].apply(lambda x: math.log(x))
#     df1[name]=df1[name].apply(lambda x: math.log(x+1))
#     df2[name]=-df2[name]
#     df=pd.concat([df1,df2])
#     df=df.sort_values(by=name)
#     y=df[name].values
#     x=range(len(y))
#     plt.scatter(x,y,c='k')
#     plt.title(name)
#     plt.savefig(r'D:\360datamining\data\firstdata\data\result\output\Outlier'+name+'.png')
#     plt.close()
#
# logtu('cost_size',out)
# logtu('current_bill_min_repay_amt',out)
# logtu('current_bill_amt',out)
# logtu('curt_jifen_max',out)
# logtu('credit_lmt_amt_max',out)