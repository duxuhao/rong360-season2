import pandas as pd
import numpy as np
from scipy.stats.mstats import mode

test = pd.read_csv('test_relation_2.txt')
train = pd.read_csv('train_relation.txt')
relation2 = pd.read_csv('relation2.txt')
relation1 = pd.read_csv('relation1.txt')
consumption_recode = pd.read_csv('consumption_recode.txt')

#for user_info
user_info = pd.read_csv('user_info.txt')
user_info = user_info[user_info.age != 'NONE']
user_info['age'] = user_info['age'].astype(float)
train = pd.concat([train[['user_id']], test[['user_id']]])
user_info = pd.merge(user_info, train[['user_id']], on = 'user_id', how = 'left',left_index = True)
user_info = user_info.fillna(-1)
X = list(user_info.columns)
X.remove('user_id')
user_info['info_num'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
for i in X:
    user_info[i+'_max'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_min'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_mean'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_std'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_sum'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_cov'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    user_info[i+'_mod'] = pd.Series(np.zeros(len(user_info)), index = user_info.index)
    
    
n = 0
for i in np.unique(user_info.user_id):
    if (n % 100 == 0):
        print n
    n += 1
    temp = user_info[user_info.user_id == i]
    user_info.loc[user_info.user_id == i, 'info_num'] = temp.shape[0]
    for tt in X:
        user_info.loc[user_info.user_id == i, tt + '_max'] = np.max(temp[tt])
        user_info.loc[user_info.user_id == i, tt + '_min'] = np.min(temp[tt])
        user_info.loc[user_info.user_id == i, tt + '_mean'] = np.mean(temp[tt])
        user_info.loc[user_info.user_id == i, tt + '_std'] = np.std(temp[tt])
        user_info.loc[user_info.user_id == i, tt + '_sum'] = np.sum(temp[tt])
        user_info.loc[user_info.user_id == i, tt + '_cov'] = float(np.cov(temp[tt]))
        user_info.loc[user_info.user_id == i, tt + '_mod'] = mode(temp[tt])[0][0]
    
user_info.to_csv('user_info_summary.csv', index = None)

#for relation1
relation1.columns = ['user_id','user2_id']
df = pd.merge(relation1, train, on = 'user_id', how = 'left')
df = df[~pd.isnull(df.lable)]
T = []
n = 0
for ind, i in enumerate(train.user_id):
    X = df[df.user_id == i]
    T.append([i, X.shape[0],train[ind:ind+1].lable.values[0]])
    if (n % 1000 == 0):
        print n
    n += 1

NN = pd.DataFrame(T)
NN.columns = ['user_id','relation1_num','lable']
NN.to_csv('train_relation1_num.csv', index = None)

relation1.columns = ['user_id','user2_id']
test = pd.read_csv('test_with_random_lable.txt')
df = pd.merge(relation1, train, on = 'user_id', how = 'left')
df = df[~pd.isnull(df.probability)]
T = []
n = 0
for ind, i in enumerate(train.user_id):
    X = df[df.user_id == i]
    T.append([i, X.shape[0],train[ind:ind+1].probability.values[0]])
    if (n % 1000 == 0):
        print n
    n += 1

NN = pd.DataFrame(T)
NN.columns = ['user_id','relation1_num','probability']
NN.to_csv('test_relation1_num.csv', index = None)

test = pd.read_csv('test_relation_2.txt')


#for relation2
relation2.columns = ['user1_id', 'user_id','relation2_type','relation2_weight','time']
re = pd.merge(relation2, train, on = 'user_id', how = 'left')
relation2.columns = ['user_id', 'user_id1','relation2_type','relation2_weight','time', 'lable2']
re.columns = ['user_id', 'user_id1','relation2_type','relation2_weight','time', 'lable2']
new = pd.merge(test, re[['user_id','relation2_type','relation2_weight', 'lable2']], on='user_id', how = 'left')
T = []
for i in np.unique(new.user_id):
    X = new[new.user_id == i]
    a = np.sum(X.lable2 * X.relation2_weight + X.relation2_type * 10000)
    if np.isnan(a):
        a = -1
    T.append([i, X.probability[:1].values[0], a])
N = pd.DataFrame(T)
N.columns = ['user_id','probability','lable2']
N[N.lable2 != -1.0].shape
N.to_csv('test_relation.txt',index = None)

#for tag
ds = pd.read_csv('rong_tag.txt')
step =20000
print 'start train'
T = []
col = ['user_id','tag_num' ,'tag_3', 'tag_4','tag_5','tag_6', 'tag_7','tag_8','tag_9']
for i in train.user_id:
    X = ds[ds.user_id == i]
    a = X.shape[0]
    temp = [i,a]
    for ind in range(300000 + step, 600000, step * 2):
        #col.append('tag_num' + str(ind))
        temp.append(X[np.abs(X.rong_tag - ind) < step].shape[0])
    T.append(temp)

N = pd.DataFrame(T)
N.columns = col
dd = pd.merge(train, N, on = 'user_id', how = 'left')
dd.to_csv('train_tag_num.csv',index = None)
print 'complete train'

print 'start test'
T = []
col = ['user_id','tag_num' ,'tag_3', 'tag_4','tag_5','tag_6', 'tag_7','tag_8','tag_9']
for i in test.user_id:
    X = ds[ds.user_id == i]
    a = X.shape[0]
    temp = [i,a]
    for ind in range(300000 + step, 600000, step * 2):
        #col.append('tag_num' + str(ind))
        temp.append(X[np.abs(X.rong_tag - ind) < step].shape[0])
    T.append(temp)

N = pd.DataFrame(T)
N.columns = col
dd = pd.merge(test, N, on = 'user_id', how = 'left')
dd.to_csv('test_tag_num.csv',index = None)
print 'complete test'

# for consumption_recode
consumption_recode = consumption_recode.ix[:, consumption_recode.columns != 'bill_id']
ColumnName = ['prior_period_bill_amt',
             'prior_period_repay_amt',
             'credit_lmt_amt',
             'curt_jifen',
             'current_bill_bal',
             'current_bill_min_repay_amt',
             'is_cheat_bill',
             'cost_cnt',
             'current_bill_amt',
             'adj_amt',
             'circle_interest',
             'prior_period_jifen_bal',
             'nadd_jifen',
             'current_adj_jifen',
             'avlb_bal_usd',
             'avlb_bal',
             'card_type',
             'pre_borrow_cash_amt_usd',
             'credit_lmt_amt_usd',
             'pre_borrow_cash_amt',
             'curr',
             'repay_stat',
             'current_min_repay_amt_usd',
             'current_repay_amt_usd',
             'current_convert_jifen',
             'current_award_jifen']

NewCol = ['user_id', 'num']
for i in ColumnName:
    #NewCol.append(i)
    NewCol.append(i+'_max')
    NewCol.append(i+'_min')
    NewCol.append(i+'_mean')
    NewCol.append(i+'_std')
    NewCol.append(i+'_sum')
    

T = []
train = pd.concat([train,test])
dftrain = pd.merge(consumption_recode, train, on = 'user_id', how = 'left',left_index = True)
dftrain = dftrain[~pd.isnull(dftrain.lable)]
print '----- looping -----'
n = 0

for i in dftrain.drop_duplicates(subset = 'user_id').user_id:
    n += 1
    if n % 1000 == 1:
        print n
    X = dftrain[dftrain.user_id == i]
    J = [i,len(X)]
    for col in ColumnName:
        J.append(np.max(X[col]))
        J.append(np.min(X[col]))
        J.append(np.mean(X[col]))
        J.append(np.std(X[col]))
        J.append(np.sum(X[col]))
    T.append(J)

df = pd.DataFrame(T)
df.columns = NewCol
df.to_csv('consumption_temp.csv', index = None)

#for tag
rong_tag = pd.read_csv('rong_tag.txt')
test = pd.read_csv('test_relation_2.txt')
train = pd.read_csv('train_relation.txt')
tag = [300014, 300021, 300028, 300056, 300098, 300119, 300154, 300196, 300231, 300301, 300357, 300385, 300427, 300462, 300469, 300497, 300525, 300539, 300553, 300567, 300574, 300595, 300658, 300672, 300686, 300735, 300749, 300763, 300770, 300777, 300903, 300931, 300994, 301015, 301036, 301057, 301085, 301106, 301176, 301183, 301211, 301253, 301260, 301302, 301309, 301330, 301414, 301547, 301568, 301638, 301666, 301687, 301701, 301729, 301848, 301946, 301981, 302016, 302100, 302107, 302240, 302282, 302324, 302359, 302471, 302688, 302716, 303717, 303759, 303885, 304039, 304151, 304312, 304417, 304634, 304753, 307448, 307679, 308372, 308470, 308659, 308799, 309758, 311284, 311865, 313314, 313818, 317129, 319572, 320370, 320657, 321441, 321756, 321798, 326446]
for t in tag:  
    train[str(t)] = pd.Series(-1 * np.ones(len(train)), index = train.index)
    test[str(t)] = pd.Series(-1 * np.ones(len(test)), index = test.index)

n = 0
train_select_ori = rong_tag.drop_duplicates(subset = 'user_id')
train_select = pd.merge(train_select_ori[['user_id']], train, on = 'user_id', how = 'left')
train_select = train_select[~pd.isnull(train_select.lable)]

for i in train_select.user_id:
    X = rong_tag[rong_tag.user_id == i]
    for t in tag:
        if t in X.rong_tag.values:
            train.loc[train.user_id == i, str(t)] = 1
        else:
            train.loc[train.user_id == i, str(t)] = 0
    n += 1
    if (n % 1000 == 0):
        print n

train.to_csv('train_relation_tag.csv', index = None)
train_select = pd.merge(train_select_ori[['user_id']], test, on = 'user_id', how = 'left')
train_select = train_select[~pd.isnull(train_select.probability)]

for i in train_select.user_id:
    X = rong_tag[rong_tag.user_id == i]
    for t in tag:
        if t in X.rong_tag.values:
            test.loc[test.user_id == i, str(t)] = 1
        else:
            test.loc[test.user_id == i, str(t)] = 0
    n += 1
    if (n % 1000 == 0):
        print n

test.to_csv('test_relation_tag.csv', index = None)