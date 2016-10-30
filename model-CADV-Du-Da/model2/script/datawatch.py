# -*- coding: utf-8 -*-
import pandas as pd
import string
import time
import numpy as np
import scipy as sp
import gc
from scipy.stats import mode
import matplotlib.pyplot as plt
userall=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_add_tag.csv')
# label=pd.read_csv(r'D:\360datamining\data\firstdata\data\train.csv')
# userall=pd.merge(user,label,on='user_id',how='inner')
#公司
input=userall[(userall.flow!=0)|(userall.business_type!=0)]
age_upper=input.age.quantile(0.75)+1.5*(input.age.quantile(0.75)-input.age.quantile(0.25))
age_lower=input.age.quantile(0.25)-1.5*(input.age.quantile(0.75)-input.age.quantile(0.25))
expect_upper=input.expect_quota.quantile(0.75)+1.5*(input.expect_quota.quantile(0.75)-input.age.quantile(0.25))
expect_lower=input.expect_quota.quantile(0.25)-1.5*(input.expect_quota.quantile(0.75)-input.age.quantile(0.25))
salary_upper=input.salary.quantile(0.75)+1.5*(input.salary.quantile(0.75)-input.age.quantile(0.25))
salary_lower=input.salary.quantile(0.25)-1.5*(input.salary.quantile(0.75)-input.age.quantile(0.25))
print age_lower,age_upper,expect_lower,expect_upper,salary_lower,salary_upper
input.to_csv(r'D:\360datamining\data\firstdata\data\user_company.csv')
print len(input),input
#个人
input2=userall[~((userall.flow!=0)|(userall.business_type!=0))]
print input2,len(input2)
input2.to_csv(r'D:\360datamining\data\firstdata\data\user_personal.csv')
# demo_data = input.loc[:,input.columns.str.contains("[a-zA-z]+")]
#
# out1=input.loc[:,['flow','expect_quota']]
# pd.scatter_matrix(out1,diagonal='bar',color='k',alpha=0.3)
# plt.show()
# plt.figure(2)
#
# plt.figure(3)
# plt.subplot(221)
# demo_data.boxplot(column=['flow'],return_type='axes')
# plt.subplot(222)
# demo_data.boxplot(column=['expect_quota'],return_type='axes')
# plt.figure(1)
# plt.subplot(221)
# demo_data.boxplot(column=['time_vary'],return_type='axes')
# plt.subplot(222)
# demo_data.boxplot(column=['age'],return_type='axes')
# plt.show()