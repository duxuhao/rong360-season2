# encoding=utf-8
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
r1=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\output\RF_union_5.csv')
r3=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\output\unionbest\RF_union_5.txt')
r1_mean=r1['lable'].mean()
r3_mean=r3['lable'].mean()
print r1_mean,r3_mean,
# r2=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\output\RF\RF_personal.csv')
r2=pd.read_csv(r'D:\360datamining\data\firstdata\data\mixture\submission_gg_6198.csv')
x1=np.array(r1['lable'])
x2=np.array(r2['probability'])
x3=np.array(r3['lable'])
# r1['lable']=1-(0.455*r3['lable']+0.545*r2['probability'])
# # r1['lable']=1-r1['lable']
# r1.to_csv(r'D:\360datamining\data\firstdata\data\mixture\mixture_1.txt',index=False)
# plt.scatter(x1,x2)
# plt.show()
print u"p系数=%f"%pearsonr(x1,x2)[0]