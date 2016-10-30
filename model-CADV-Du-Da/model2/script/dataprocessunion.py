# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import math
start=time.clock()
#散布图矩阵绘图函数
def scattermatrix(features):#feature is list
    out1=out.loc[:,features]
    plt.title(u"散布图矩阵")
    pd.scatter_matrix(out1,diagonal='bar',color='k',alpha=0.3)
    plt.savefig(r'D:\360datamining\data\firstdata\data\result\output\graph\scattermatrix.png')
#离群点检测绘图函数#
def  loggraph(name,df):
    df1=df[df[name]>=0]
    df2=df[df[name]<0]
    df2[name]=-df2[name]
    df2[name]=df2[name].apply(lambda x: math.log(int(x)))
    df1[name]=df1[name].apply(lambda x: math.log(int(x)+1))
    df2[name]=-df2[name]
    df=pd.concat([df1,df2])
    df=df.sort_values(by=name)
    y=df[name].values
    x=range(len(y))
    plt.scatter(x,y,c='k')
    plt.title(name)
    plt.savefig(r'D:\360datamining\data\firstdata\data\result\output\graph\Outlier'+name+'.png')
    plt.close()
def boxplt(df,feature1,feature2):#df is dataframe /// feature1 and feature2 are lists
    demo_data = df.loc[:,df.columns.str.contains("[a-zA-z]+")]
    plt.figure(1)
    plt.title('user-info-boxplot')
    plt.subplot(211)
    demo_data.boxplot(column=feature1,return_type='axes')#'expect_quota','salary','flow','gross_profit',
    plt.subplot(212)
    demo_data.boxplot(column=feature2,return_type='axes')#['sex','occupation','education','marital_status','live_info','local_hk','money_function','company_type','school_type','business_type','business_year','personnel_num','pay_type','time_vary']
    plt.savefig(r'D:\360datamining\data\firstdata\data\result\output\graph\user-info-boxplot.png')

################################################
#####数据处理一期----用户基本数据处理###########
################################################
df1=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_info.csv')#原始user-info表保存路径
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
#数据暂存
# df3.to_csv(r'D:\360datamining\data\firstdata\data\user_info_tf1.csv',index=False)#暂存表1
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
#添加特征
df5=df4.groupby('user_id').agg('max').reset_index()
df5['size']=pd.Series(tuple(user_size))
size=np.array(tuple(user_size))
ep_qta=np.array(df5['expect_quota'])
df5['my_sum']=pd.Series(ep_qta*size)
df5['time_vary']=pd.Series(time_vary)
df5['time_mean']=pd.Series(time_mean)
df5['tm_min']=pd.Series(time_min)
print u"数据一期处理结束"
#####################################################
#############数据处理二期----社交表1信息处理#########
#####################################################
relation1=pd.read_csv(r'D:\360datamining\data\firstdata\data\relation1.csv')#原始relation1表保存路径
user=df5
user_list=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\userlist.csv')#训练集+验证集所有用户id保存路径
relationtemp1=pd.DataFrame(columns=['user_id','relation1_num'])
relationtemp1.user_id=relation1.groupby('user1_id').count().index
relationtemp1.relation1_num=relation1.groupby('user1_id').count().values
relation=pd.merge(relationtemp1,user_list,on='user_id',how='inner')
concat=pd.merge(user,relation,on='user_id',how='left')
##############填充缺失值############
fill_mode=lambda g: g.fillna(mode(g).mode[0])
concat.fillna(concat.median(),inplace=True)
print concat.head(10)
print u"数据二期处理结束"
#####################################################
################数据处理三期——数据清洗/绘图########
#####################################################
input=concat
#绘制图
loggraph('salary',input)
loggraph('expect_quota',input)
loggraph('age',input)
loggraph('relation1_num',input)
loggraph('size',input)
loggraph('tm_encode',input)
loggraph('my_sum',input)
#异常值替换
agein=input.age.median()
salaryin=input.salary.quantile(0.5)
expin=input.expect_quota.quantile(0.5)
sumin=input.my_sum.quantile(0.5)
relationin=input.relation1_num.quantile(0.5)
input.to_csv(r'D:\360datamining\data\firstdata\data\result\output\maybe.csv')#暂存表2//方便单步执行
####人工规则清洗数据#####
for i in range(0,38261):
    if (input.iloc[:,1][i]<math.exp(3))|(input.iloc[:,1][i]>math.exp(4.1)):
        input.iloc[:,1][i]=agein
    if (input.iloc[:,6][i]<math.exp(4.5))|(input.iloc[:,6][i]>math.exp(11.5)):
        input.iloc[:,6][i]=expin
    if (input.iloc[:,18][i]>15000)|(input.iloc[:,18][i]<math.exp(4.5)):
        input.iloc[:,18][i]=salaryin

input.to_csv(r'D:\360datamining\data\firstdata\data\temp.csv')#暂存表3//方便单步执行
print u"数据三期处理结束"
######################################################
###############数据处理四期--消费数据#################
######################################################
usercost1=pd.read_csv(r'D:\360datamining\data\firstdata\data\consumption_recode.csv')#原始消费信息表保存路径
user2=pd.read_csv(r'D:\360datamining\data\firstdata\data\temp.csv')#读取暂存表
######消费数据###############
cost=pd.DataFrame(columns=['user_id','cost_size','credit_lmt_amt','curt_jifen','current_bill_bal','current_bill_min_repay_amt','current_bill_amt','credit_lmt_amt_max','curt_jifen_max','prior_period_bill_amt','prior_period_repay_amt','real_pay_max','real_pay_std','prior_period_bill_amt_max','prior_period_repay_amt_max','credit_lmt_amt_std','curt_jifen_std','real_pay_size','pre_borrow_cash_amt'])
cost.user_id=usercost1.groupby('user_id').count().index
cost.cost_size=usercost1.groupby('user_id').count().values
cost.credit_lmt_amt=usercost1.groupby('user_id')['credit_lmt_amt'].median().values
cost.curt_jifen=usercost1.groupby('user_id')['curt_jifen'].median().values
cost.current_bill_bal=usercost1.groupby('user_id')['current_bill_bal'].median().values
cost.current_bill_min_repay_amt=usercost1.groupby('user_id')['current_bill_min_repay_amt'].median().values
cost.current_bill_amt=usercost1.groupby('user_id')['current_bill_amt'].median().values
cost.credit_lmt_amt_max=usercost1.groupby('user_id')['credit_lmt_amt'].max().values
cost.curt_jifen_max=usercost1.groupby('user_id')['curt_jifen'].max().values
cost.credit_lmt_amt_std=usercost1.groupby('user_id')['credit_lmt_amt'].std().values
cost.curt_jifen_std=usercost1.groupby('user_id')['curt_jifen'].std().values
cost.prior_period_bill_amt=usercost1.groupby('user_id')['prior_period_bill_amt'].median().values
cost.prior_period_repay_amt=usercost1.groupby('user_id')['prior_period_repay_amt'].median().values
cost.prior_period_bill_amt_max=usercost1.groupby('user_id')['prior_period_bill_amt'].max().values
cost.prior_period_repay_amt_max=usercost1.groupby('user_id')['prior_period_repay_amt'].max().values
cost.real_pay_max=usercost1.groupby('user_id')['prior_period_bill_amt'].max().values-usercost1.groupby('user_id')['prior_period_repay_amt'].max().values
cost.real_pay_std=usercost1.groupby('user_id')['prior_period_bill_amt'].std().values-usercost1.groupby('user_id')['prior_period_repay_amt'].std().values
cost.real_pay_size=cost.cost_size*cost.real_pay_max
cost.pre_borrow_cash_amt=usercost1.groupby('user_id')['pre_borrow_cash_amt'].max().values
out=pd.merge(user2,cost,on='user_id',how='left')
##缺失值填充
out['cost_size']=out['cost_size'].replace(np.nan,16.0)
out['credit_lmt_amt']=out['credit_lmt_amt'].replace(np.nan,610928.998009)
out['curt_jifen']=out['curt_jifen'].replace(np.nan,out.curt_jifen.median())
out['current_bill_bal']=out['current_bill_bal'].replace(np.nan,out.current_bill_bal.median())
out['current_bill_min_repay_amt']=out['current_bill_min_repay_amt'].replace(np.nan,out.current_bill_min_repay_amt.median())
out['current_bill_amt']=out['current_bill_amt'].replace(np.nan,out.current_bill_amt.median())
out['credit_lmt_amt_max']=out['credit_lmt_amt_max'].replace(np.nan,out.credit_lmt_amt_max.median())
out['curt_jifen_max']=out['curt_jifen_max'].replace(np.nan,out.curt_jifen_max.median())
out['credit_lmt_amt_std']=out['credit_lmt_amt_std'].replace(np.nan,out.credit_lmt_amt_std.median())
out['curt_jifen_std']=out['curt_jifen_std'].replace(np.nan,out.curt_jifen_std.median())
out['prior_period_bill_amt']=out['prior_period_bill_amt'].replace(np.nan,out.prior_period_bill_amt.median())
out['prior_period_repay_amt']=out['prior_period_repay_amt'].replace(np.nan,out.prior_period_repay_amt.median())
out['prior_period_bill_amt_max']=out['prior_period_bill_amt_max'].replace(np.nan,out.prior_period_bill_amt_max.median())
out['prior_period_repay_amt_max']=out['prior_period_repay_amt_max'].replace(np.nan,out.prior_period_repay_amt_max.median())
out['real_pay_max']=out['real_pay_max'].replace(np.nan,out.real_pay_max.median())
out['real_pay_std']=out['real_pay_std'].replace(np.nan,out.real_pay_std.median())
out['real_pay_size']=out['real_pay_size'].replace(np.nan,out.real_pay_std.median())
out['pre_borrow_cash_amt']=out['pre_borrow_cash_amt'].replace(np.nan,out.pre_borrow_cash_amt.median())
out.to_csv(r'D:\360datamining\data\firstdata\data\result\uniontest2.csv')#模型输入特征表
print u"数据四期处理结束"
#变量绘图


