from scipy.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from collections import OrderedDict
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from multiprocessing import Pool
import warnings

def k_fold(df, step, selectcol, clf, auc, tempx, num, addfeature, rem, coetest=0):
    totaltest = []
    selectcol = list(OrderedDict.fromkeys(selectcol))
    clf1 = xgb.XGBClassifier(n_estimators=125, max_depth = 3, learning_rate = 0.05)
    X = df[selectcol]
    y = df.lable
    cv = StratifiedKFold(y, n_folds=3) #use three fold
    for train, test in cv:
        totaltest.append(roc_auc_score(y[test], clf1.fit(X.ix[train,:], y[train]).predict_proba(X.ix[test,:])[:,1]))
    if np.mean(totaltest) > auc:
        cc = []
        coltemp = selectcol[:]
        coltemp.remove(addfeature)
        for xx in coltemp:
            cc.append(pearsonr(df[addfeature],df[xx])[0]) #check the correlation coefficient
        if (np.abs(np.max(cc)) < 0.95) | coetest:
            f = open('log_LRS_SA_RGSS.txt','a') #record all the imporved combination
            f.write(num)
            f.write('  ')
            f.write(addfeature)
            f.write('  ')
            f.write(str(np.abs(np.max(cc))))
            f.write(': \n')
            f.write(str(np.round(np.mean(totaltest),6)))
            f.write('\t+\t')
            f.write(str(selectcol[:]))
            f.write('\n')
            f.write('*-' * 50)
            f.write('\n')
            f.close()
            tempx = selectcol[:]
            auc = np.mean(totaltest)
            rem = addfeature
    return tempx, auc, rem

def newdataset(df, feature):
    for i in feature:
        newfeature = i.split(' ')
        if len(newfeature) == 1:
            pass
        elif newfeature[1] == '*':
            df[i] = pd.Series(df[newfeature[0]] * df[newfeature[2]], index = df.index)
        elif newfeature[1] == '/':
            df[i] = pd.Series((df[newfeature[0]]+0.01) / (df[newfeature[2]]+0.01), index = df.index)
        elif newfeature[1] == '+':
            df[i] = pd.Series(np.array(normalize(df[[newfeature[0]]], axis = 0) + normalize(df[[newfeature[2]]], axis = 0)).ravel(), index = df.index)
        elif newfeature[1] == '-':
            df[i] = pd.Series(np.array(normalize(df[[newfeature[0]]], axis = 0) - normalize(df[[newfeature[2]]], axis = 0)).ravel(), index = df.index)
    return df

def greedy(clf, df, auc, Startcol, tempx, col, step = 3000, remain = ''):
    print '-' * 50 + 'start greedy' + '-' * 50
    for i in tempx:
        col.remove(i)
    PotentialAdd = ['tm_encode_max','300028_301687', 'education_max_info_num','300196_2','300196','300028','relation1_num','cost_cnt_mean','prior_period_repay_amt_max','circle_interest_max','pre_borrow_cash_amt_mean','current_bill_amt_mean','expect_quota_min','current_adj_jifen_max','avlb_bal_max','prior_period_jifen_bal_std','time_gap','credit_lmt_amt_max','is_cheat_bill_std','credit_lmt_amt_usd_mean','current_bill_bal_mean']
    dele = ''
    bestauc = auc
    bestfeature = tempx
    while (Startcol != tempx) | (PotentialAdd != []): #stop when no improve for the last round and no potential add feature
        if Startcol == tempx:
            if auc > bestauc:
                bestauc = auc
                bestfeature = tempx
            auc += -0.0005 #Simulate Anneal Arithmetic
            tempx.append(PotentialAdd[0])
        print '*' * 20,
        print ' ' + str(len(tempx)+1) + ' round ',
        print '*' * 20
        
        if remain in col:
            col.remove(remain)
        if dele != '':
            print 'remove: ' + dele
            col.append(dele)
        dele = ''    
        Startcol = tempx[:]
        #forward
        for sub, i in enumerate(col[:]): 
            selectcol = Startcol[:]
            selectcol.append(i)
            tempx, auc, remain = k_fold(df, step, selectcol, clf, auc, tempx, str(1+sub), i, remain) 
        
        #backward
        for i in tempx[:-2]:
            deletecol = tempx[:]
            if i in deletecol:
                deletecol.remove(i)
            tempx, auc, dele = k_fold(df, step, deletecol, clf, auc, tempx, 'reverse', i, dele, 1)
        for i in tempx:
            if i in PotentialAdd:
                PotentialAdd.remove(i)
    print '-' * 20 + 'complete greedy' + '-' * 20
    if auc > bestauc:
        bestauc = auc
        bestfeature = tempx
    return bestfeature, bestauc

def myrandom(clf, df, auc, Startcol, col, step = 3000):
    print '-' * 20 + 'start random' + '-' * 20
    good = Startcol[:]
    for i in Startcol:
        col.remove(i)
    for t in range(4,8):
        print 'add ' + str(t) + ' features'
        for i in range(50):
            selectcol = random.sample(col, t)
            for add in Startcol: 
                selectcol.append(add)
            good, randomauc, rem = k_fold(df, step, selectcol, clf, auc, good, str(i), 'None', '')
    print '-' * 20 + 'complete random' + '-' * 20
    return good, randomauc

def LRS_SA_RGSS_combination(randomauc, df, Startcol, tempx, ColumnName):
    auc = 0
    clf = 0
    while randomauc > auc:
        Startcol, auc = greedy(clf, df, randomauc, Startcol, tempx, ColumnName[:])
        print 'random select starts with',
        print Startcol
        Startcol, randomauc = myrandom(clf, df, auc, Startcol, ColumnName[:])
        tempx = Startcol[:]
    print '*-*' * 50
    print 'best auc:' + str(auc)
    print 'best features combination: ',
    print tempx
    return tempx

def obtaincol(df, delete):
    ColumnName = list(df.columns)
    for i in delete:
        if i in ColumnName:
            ColumnName.remove(i)
    return ColumnName

def main(temp):
    train = pd.read_csv('train_new_label_2.csv')
    set = pd.read_csv('xgbimpute.csv')
    df = pd.merge(train[['user_id','lable']], set, on = 'user_id', how = 'left', left_index = True)
    df = df.fillna(-100)
    df = newdataset(df, temp[:])
    ColumnName = obtaincol(df, ['user_id','lable','lable2']) #obtain columns withouth the useless features
    Startcol = LRS_SA_RGSS_combination(0.1, df, ['None'], temp, ColumnName[:])
    print Startcol

if __name__ == "__main__":
    pool = Pool(8)
    temp = ['tm_encode_max']
    main(temp)
