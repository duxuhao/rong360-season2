# encoding=utf-8
from scipy import interp
import pandas as pd
# user=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_add_cost.csv')
# print user.head(5)
# del user['Unnamed: 0'],
# del user['Unnamed: 0.1'],
#
# del user['Unnamed: 0.1.1']
# # del user['Unnamed: 0.1'],
# stats= user.select_dtypes(include=['O','float64','int64']).describe().T\
#     .assign(missing_pct=user.apply(lambda x : float(x[x==0].count())/len(x)))
# stats.to_excel(r'D:\360datamining\data\firstdata\data\user_add_cost.xlsx')
# help (interp)
# # pipeline=SVC(kernel='sigmoid',class_weight='balanced',probability=True)
# # polynomial_features = PolynomialFeatures(degree=2,
# #                                              include_bias=False)
# # pipeline=Pipeline([('polynomial_features',polynomial_features),('RandomForestClassifier',model)])
# # #KNN
# # # model=KNeighborsClassifier(n_neighbors=27)
# # #SVM
# # # model= svm.SVC(kernel='sigmoid', probability=True,)
# # #逻辑回归
# # # model = LogisticRegression()
# help(pd.fillna())

data=[]
i=15
for m in range(0,3):
    for j in range(0,6):
        data.append(['2016-09-14-%i-%i'%(i,j)])
        if j==5:
            i+=1
    m+=1


print data

