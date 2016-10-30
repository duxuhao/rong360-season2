# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score#####交叉验证########
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc,roc_auc_score
from scipy import interp
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
########开始#########i
start=time.clock()
user=pd.read_csv(r'uniontest2.csv')#模型输入表
label=pd.read_csv(r'train.csv')#原始train表保存路径
traindata=pd.merge(user,label,on='user_id',how='inner')

print u'数据读取成功'

############输入特征############
features=['tm_encode','salary','live_info','local_hk','size','relation1_num','expect_quota','time_mean','cost_size','my_sum','credit_lmt_amt_max','curt_jifen_max','prior_period_bill_amt','prior_period_repay_amt','occupation', 'marital_status','education',]
##################训练数据#######################
traindata_X=traindata.loc[:,features]
t= traindata_X.corr()
traindata_y=traindata.loc[:,['lable']]
##################无量纲化###################
maxmin=StandardScaler().fit(traindata_X)
data_X=maxmin.transform(traindata_X)
traindata_X=pd.DataFrame(data_X,index=traindata_X.index)
print u'无量纲化结束'

##############模型训练####ROC###########
names =features
##交叉验证
foldnum = 15 #fold times
skf = StratifiedKFold(traindata_y.values.ravel(), n_folds=foldnum)
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
mean_tpr = 0.0
auc_gini=[]
auc_entropy=[]
#随机森林,gini+entropy
pipeline=RandomForestClassifier(n_estimators=2000,bootstrap=True,min_samples_leaf=51,max_depth=38,n_jobs=-1,class_weight='balanced_subsample',max_features='sqrt',criterion='gini')
pipeline1=RandomForestClassifier(n_estimators=2000,bootstrap=True,min_samples_leaf=51,max_depth=38,n_jobs=-1,class_weight='balanced_subsample',max_features='sqrt',criterion='entropy')

# # 交叉验证分层采样绘制ROC
for i, (train, test) in enumerate(skf):
    traindt_X,traindt_y=traindata_X[traindata_X.index.isin(train)], traindata_y[traindata_X.index.isin(train)].values.ravel()
    testd_X,testd_y=traindata_X[traindata_X.index.isin(test)], traindata_y[traindata_X.index.isin(test)].values.ravel()
    print (u'第 %s 次完成'%i)
    pipeline.fit(traindt_X, traindt_y)
    pipeline1.fit(traindt_X, traindt_y)
    #save model#模型保存路径
    joblib.dump(pipeline,'RF_gini_%d.pkl'%i,compress=3)
    joblib.dump(pipeline,'RF_entropy_%d.pkl'%i,compress=3)
    probas_=pipeline.predict_proba(testd_X)
    probas1_=pipeline1.predict_proba(testd_X)
    y_pred=pipeline.predict(testd_X)
    y_pred1=pipeline1.predict(testd_X)
    # print(u"gini_oobscore:%f"%pipeline.oob_score_)
    # print(u"entropy_oobscore:%f"%pipeline1.oob_score_)
    auc_gini.append(roc_auc_score(testd_y,probas_[:, 1]))
    auc_entropy.append(roc_auc_score(testd_y,probas1_[:, 1]))
    print (u"rocaucscore_1:%f"%roc_auc_score(testd_y,probas_[:, 1]))
    print(u"rocaucscore_2:%f"%roc_auc_score(testd_y,probas1_[:, 1]))
    print(u'F1分数gini：%s'%f1_score(testd_y,y_pred))
    print(u'F1分数entropy：%s'%f1_score(testd_y,y_pred1))
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(testd_y, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
# print(u"obb_mean:%f"%np.mean(obb_mean))
# print(u"obb_entropy_mean:%f"%np.mean(obb_mean_entropy))
print(u"auc_gini:%f"%np.mean(auc_gini))
print(u"auc_entropy:%f"%np.mean(auc_entropy))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr /= len(skf)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
'''
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
print u'交叉验证结束'
end=time.clock()
print ("runtime:%s Seconds"%(end-start))
###################测试数据######################
testdt=pd.read_csv(r'test.csv')#原始test表保存路径
test=pd.merge(testdt,user,on='user_id',how='inner')
test1=pd.merge(testdt,user,on='user_id',how='inner')
union=pd.merge(testdt,user,on='user_id',how='inner')
#查看test数据0值占比#需删掉read-csv可能产生（Unnamed：）列
# testtable=test.select_dtypes(include=['O','float64','int64']).describe().T\
#     .assign(missing_pct=test.apply(lambda x : float(x[x==0].count())/len(x)))
# testtable.to_excel(r'testtable.xlsx')
#无量纲化
testdata_X=test.loc[:,features]
datats_X=maxmin.transform(testdata_X)
testdata_X=pd.DataFrame(datats_X,index=testdata_X.index)
####################输出结果#####################
#读取模型
output=[]
output1=[]
for i in range(0,foldnum):
    clf = joblib.load(r'RF_gini_%d.pkl'%i)
    clf1 = joblib.load(r'RF_entropy_%d.pkl'%i)
    print (u'gini特征重要性排序：第%i次'%i)
    print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
                 names), reverse=True)
    print (u'entropy特征重要性排序：第%i次'%i)
    print sorted(zip(map(lambda x: round(x, 4), clf1.feature_importances_),
                 names), reverse=True)
    model_probs=clf.predict_proba(testdata_X)
    model_probs1=clf1.predict_proba(testdata_X)
    model_probs2Series=[]
    model_probs2Series1=[]
    for list in  model_probs:
        model_probs2Series.append(list[1])
    for list in  model_probs1:
        model_probs2Series1.append(list[1])

    output.append(model_probs2Series)
    output1.append(model_probs2Series1)
    test['lable']=pd.Series(model_probs2Series)
    test1['lable']=pd.Series(model_probs2Series1)
    union['lable']= 0.51*test['lable']+0.49*test1['lable']
    result=pd.DataFrame({'user_id':test['user_id'],'lable':test['lable']}).reindex(columns=['user_id','lable'])
    result.to_csv(r'RF_final_gini_%i.csv'%i,index=False)#模型-gini
    result1=pd.DataFrame({'user_id':test1['user_id'],'lable':test1['lable']}).reindex(columns=['user_id','lable'])
    result1.to_csv(r'RF_final_entropy_%i.csv'%i,index=False)#模型-entropy
    union=pd.DataFrame({'user_id':union['user_id'],'lable':union['lable']}).reindex(columns=['user_id','lable'])
    union.to_csv(r'RF_union_%i.csv'%i,index=False)#模型加权融合
output=np.array(output)
output1=np.array(output1)
print u'矩阵化结果%s'%output
print u'矩阵化结果entropy%s'%output1
#结果输出
output=np.mean(output,axis=0)
output1=np.mean(output1,axis=0)
outputunion=0.51*output+0.49*output1
print u'求均值结果%s'%output
print u'求均值结果entropy%s'%output1
test['lable']=pd.Series(output)
test1['lable']=pd.Series(output1)
union['lable']=0.51*test['lable']+0.49*test1['lable']
result_mean=pd.DataFrame({'user_id':test['user_id'],'lable':test['lable']}).reindex(columns=['user_id','lable'])
result_mean.to_csv(r'RF_personal.csv',index=False)#模型-gini
result_mean1=pd.DataFrame({'user_id':test1['user_id'],'lable':test1['lable']}).reindex(columns=['user_id','lable'])
result_mean1.to_csv(r'RF_entropy_personal.csv',index=False)#模型-entropy
unionmean=pd.DataFrame({'user_id':union['user_id'],'lable':union['lable']}).reindex(columns=['user_id','lable'])
unionmean.columns = ['user_id','probability']
unionmean.to_csv(r'../../model2_answer.txt',index=False)#模型加权融合
print u'结束'