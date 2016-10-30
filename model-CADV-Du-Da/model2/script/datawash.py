# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import math
input=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_add_cost.csv')
print input.head(5)


def  logtu(name,df):
    df1=df[df[name]>=0]
    df2=df[df[name]<0]
    df2[name]=-df2[name]
    df2[name]=df2[name].apply(lambda x: math.log(x))
    df1[name]=df1[name].apply(lambda x: math.log(x+1))
    df2[name]=-df2[name]
    df=pd.concat([df1,df2])
    df=df.sort_values(by=name)
    y=df[name].values
    x=range(len(y))
    plt.scatter(x,y,c='k')
    plt.title(name)
    plt.savefig(r'D:\360datamining\data\firstdata\data\result\output\Outlier'+name+'afterwash.png')
    plt.close()

# logtu('salary',input)
# logtu('expect_quota',input)
# logtu('age',input)
# logtu('relation1_num',input)
# logtu('size',input)
# logtu('tm_encode',input)
# logtu('my_sum',input)
# agein=input.age.quantile(0.5)
# salaryin=input.salary.quantile(0.5)
# expinhigh=input.expect_quota.quantile(0.5)
# expinlow=input.expect_quota.quantile(0.5)
# sumin=input.my_sum.quantile(0.5)
# relationin=input.relation1_num.quantile(0.5)
credin=input.credit_lmt_amt.quantile(0.5)
curtin=input.curt_jifen.quantile(0.5)

print credin,curtin
# # # Artificial rules


# for i in range(0,38261):
    # if input.iloc[:,4][i]<math.exp(3):
    #     input.iloc[:,4][i]=agein#0.6
    # if input.iloc[:,4][i]>math.exp(4.1):
    #     input.iloc[:,4][i]=agein#0.75
    # if input.iloc[:,9][i]<math.exp(4.5):
    #     input.iloc[:,9][i]=expinlow
    # if input.iloc[:,9][i]>math.exp(11.5):
    #     input.iloc[:,9][i]=expinhigh#0.8
    #
    # if (input.iloc[:,21][i]>15000)|(input.iloc[:,21][i]<math.exp(4.5)):
    #     input.iloc[:,21][i]=salaryin#0.5
    # if (input.iloc[:,25][i]>math.exp(16.6))|(input.iloc[:,25][i]<math.exp(5)):
    #     input.iloc[:,25][i]=sumin
    # if (input.iloc[:,24][i]>math.exp(4))|(input.iloc[:,24][i]<math.exp(1.7)):
    # #     input.iloc[:,24][i]=int(math.exp(1.9))
    # if (input.iloc[:,30][i]>math.exp(9))|(input.iloc[:,30][i]<math.exp(1)):
    #     input.iloc[:,30][i]=relationin
    ####新加/未修改
    # if ((abs(input.iloc[:,37][i])<math.exp(7.5))and(abs(input.iloc[:,37][i])>=0))or(input.iloc[:,37][i]<-math.exp(10)):
    #     input.iloc[:,37][i]=credin#0.5
    # if ((abs(input.iloc[:,38][i])<1)and(abs(input.iloc[:,38][i])>0))or((input.iloc[:,38][i])<=-math.exp(5)):
    #     input.iloc[:,38][i]=curtin
print input.head(5)
logtu('curt_jifen',input)
logtu('credit_lmt_amt',input)
input.to_csv(r'D:\360datamining\data\firstdata\data\user_cost_wash.csv')


# demo_data = input.loc[:,input.columns.str.contains("[a-zA-z]+")]
# plt.figure(2)
# demo_data.boxplot(column=['salary'],return_type='axes')
# plt.figure(3)
# demo_data.boxplot(column=['expect_quota'],return_type='axes')
# plt.figure(1)
# plt.subplot(211)
# demo_data.boxplot(column=['time_vary'],return_type='axes')
# plt.subplot(212)
# demo_data.boxplot(column=['live_info','local_hk','age'],return_type='axes')
# plt.show()