from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
import random
import sys
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from multiprocessing import Pool
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

def select(x,df,aucfix, name = 'nolabel'):
        auctest = []
        X = x[:]
        y = df[:].lable
        clf1 = xgb.XGBClassifier(n_estimators=125, max_depth = 3, learning_rate = 0.05)
        cv = StratifiedKFold(y, n_folds=3)
        for train, test in cv:
            auctest.append(roc_auc_score(y[test], clf1.fit(X.ix[train,:], y[train]).predict_proba(X.ix[test,:])[:,1]))
        a = np.mean(auctest)
        selectcol = list(x.columns)
        selectcol.remove(name)
        if a > aucfix:
            nn = []
            for t in selectcol:
                nn.append(np.abs(pearsonr(x[name], x[t])[0]))
            f = open('combine_6195_6927.txt','a')
            f.write('-'*50)
            f.write('\n')
            f.write(name)
            f.write('\n')
            f.write('max coeeficient:')
            f.write(str(max(nn)))
            f.write('\n')
            f.write('test: ')
            f.write(str((np.round(a,8) - aucfix)*1000))
            f.write('\t')
            f.write(str(np.round(a,8)))
            f.write('\n')
            f.write(str(list(x.columns)))
            f.write('\n')
            f.write('-'*50)
            f.write('\n')
            f.close()

def addcolumns(feature1,feature2,df,x,n):
    if n == 0:
        name = feature1 + ' * ' + feature2
        x[name] = pd.Series(df[feature1] * df[feature2], index = df.index)
        return x, name
    elif n == 1:
        name = feature1 + ' / ' + feature2
        x[name] = pd.Series((df[feature1]+0.01) / (df[feature2]+0.01), index = df.index)
        return x, name
    elif n == 2:
        name = feature1 + ' + ' + feature2
        x[name] = pd.Series(np.array(normalize(df[[feature1]], axis = 0) + normalize(df[[feature2]], axis = 0)).ravel(), index = df.index)
        return x, name
    elif n == 3:
        name = feature1 + ' - ' + feature2
        x[name] = pd.Series(np.array(normalize(df[[feature1]], axis = 0) - normalize(df[[feature2]], axis = 0)).ravel(), index = df.index)
        return x, name
    elif n == 4:
        name = feature1
        x[name] = pd.Series(df[feature1], index = df.index)
        return x, name

def add_combination(df):
    df['expect_quota_mod - current_repay_amt_usd_min / salary_max']  = pd.Series((np.array(normalize(df[['expect_quota_mod']], axis = 0) - normalize(df[['current_repay_amt_usd_min']], axis = 0)).ravel()+0.01) / (df['salary_max']+0.01), index = df.index)
    df['prior_period_bill_amt_mean * marital_status_min'] = pd.Series(df['prior_period_bill_amt_mean'] * df['marital_status_min'], index = df.index)
    df['nolabel + prior_period_bill_amt_mean * marital_status_min'] = pd.Series(np.array(normalize(df[['nolabel']], axis = 0) + normalize(df[['prior_period_bill_amt_mean * marital_status_min']],axis = 0)).ravel(), index = df.index)
    df['tm_encode_max / prior_period_repay_amt_std'] = pd.Series((df['tm_encode_max']+0.01) / (df['prior_period_repay_amt_std']+0.01), index = df.index)
    df['money_function_mean / tm_encode_max / prior_period_repay_amt_std'] = pd.Series((df['money_function_mean']+0.01) / (df['tm_encode_max / prior_period_repay_amt_std']+0.01), index = df.index)
    df['money_function_mean * current_award_jifen_max'] = pd.Series(df['money_function_mean'] * df['current_award_jifen_max'], index= df.index)
    df['money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std'] = pd.Series(df['money_function_mean * current_award_jifen_max'] * df['tm_encode_max / prior_period_repay_amt_std'], index= df.index)
    df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series(np.array(normalize(df[['current_bill_amt_std']], axis = 0) - normalize(df[['expect_quota_mod - current_repay_amt_usd_min / salary_max']], axis = 0)).ravel(), index = df.index)
    df['time_gap * pay_type'] = pd.Series(df['time_gap'] * df['pay_type'], index = df.index)
    df['occupation_std / marital_status'] = pd.Series((df['occupation_std']+0.01) / (df['marital_status']+0.01), index = df.index)
    df['time_gap * pay_type - occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['time_gap * pay_type']], axis = 0) - normalize(df[['occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['tag_num * 300196'] = pd.Series(df['tag_num'] * df['300196'], index = df.index)
    df['tag_num * 300196 - occupation_std'] = pd.Series(np.array(normalize(df[['tag_num * 300196']], axis = 0) - normalize(df[['occupation_std']], axis = 0)).ravel(), index = df.index)
    df['tm_encode_max / time_gap * pay_type'] = pd.Series((df['tm_encode_max'] + 0.01) / (df['time_gap * pay_type'] + 0.01), index = df.index)
    df['current_award_jifen_max * expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(np.array(normalize(df[['expect_quota_mod']], axis = 0) - normalize(df[['current_repay_amt_usd_min']], axis = 0)).ravel() * df['current_award_jifen_max'], index = df.index)
    df['expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(np.array(normalize(df[['expect_quota_mod']], axis = 0) - normalize(df[['current_repay_amt_usd_min']], axis = 0)).ravel(), index = df.index)
    df['expect_quota_mod - current_repay_amt_usd_min + salary_max'] = pd.Series(np.array(normalize(df[['expect_quota_mod - current_repay_amt_usd_min']], axis = 0) + normalize(df[['salary_max']], axis = 0)).ravel(), index = df.index)
    df['tm_encode_max / time_gap * pay_type'] = pd.Series((df['tm_encode_max']+0.01) / (df['time_gap * pay_type']+0.01),index = df.index)
    df['tag_num * 300196 / current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series((df['tag_num * 300196']+0.01) / (df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max']+0.01), index = df.index)
    df['time_gap * pay_type - occupation_std / marital_status * 301211'] = pd.Series(df['time_gap * pay_type - occupation_std / marital_status'] * df['301211'], index = df.index)
    df['tm_encode_max / time_gap * pay_type + occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['tm_encode_max / time_gap * pay_type']], axis = 0) + normalize(df[['occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * adj_amt_min'] = pd.Series(df['education_max_info_num'] * df['adj_amt_min'], index = df.index)
    df['education_max_info_num * adj_amt_min - tm_encode_max / time_gap * pay_type + occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['education_max_info_num * adj_amt_min']], axis = 0) - normalize(df[['tm_encode_max / time_gap * pay_type + occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num'] = pd.Series(np.array(normalize(df[['money_function_mean / tm_encode_max / prior_period_repay_amt_std']], axis = 0) + normalize(df[['relation1_num']], axis = 0)).ravel(), index = df.index)
    df['prior_period_bill_amt_mean * marital_status_min + education_max_info_num * tag_6'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean * marital_status_min']], axis = 0) + normalize(df[['time_gap * pay_type']], axis = 0)).ravel(), index = df.index)
    df['tag_num / relation1_num'] = pd.Series((df['tag_num'] + 0.01) / (df['relation1_num'] + 0.01), index = df.index)
    df['info_num / tag_num / relation1_num'] = pd.Series((df['info_num'] + 0.01) / (df['tag_num / relation1_num'] + 0.01), index = df.index)
    df['tag_num * 300196 / expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series((df['tag_num * 300196']+0.01) / (df['expect_quota_mod - current_repay_amt_usd_min / salary_max']+0.01), index = df.index)
    df['circle_interest_max * occupation_std / marital_status'] = pd.Series(df['circle_interest_max'] * df['occupation_std / marital_status'], index = df.index)
    df['prior_period_bill_amt_mean + time_gap * pay_type'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean']], axis = 0) + normalize(df[['time_gap * pay_type']], axis = 0)).ravel(), index = df.index)
    df['tm_encode_max / prior_period_repay_amt_std - time_gap'] = pd.Series(np.array(normalize(df[['tm_encode_max / prior_period_repay_amt_std']], axis = 0) - normalize(df[['time_gap']], axis = 0)).ravel(), index = df.index)
    df['relation1_num + money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num'] = pd.Series(np.array(normalize(df[['relation1_num']], axis = 0) + normalize(df[['money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num']], axis = 0)).ravel(), index = df.index)
    df['max_month_repay * prior_period_bill_amt_mean'] = pd.Series(df['max_month_repay'] * df['prior_period_bill_amt_mean'], index = df.index)
    df['education_max_info_num * adj_amt_min * max_month_repay * prior_period_bill_amt_mean'] = pd.Series(df['education_max_info_num * adj_amt_min'] * df['max_month_repay * prior_period_bill_amt_mean'], index = df.index)
    df['expect_quota * money_function_mean * current_award_jifen_max'] = pd.Series(df['expect_quota'] * df['money_function_mean * current_award_jifen_max'],index = df.index)
    df['expect_quota * education_max_info_num * adj_amt_min'] = pd.Series(df['expect_quota'] * df['education_max_info_num * adj_amt_min'],index=df.index)
    df['nolabel * tag_num / relation1_num'] = pd.Series(df['nolabel'] * df['tag_num / relation1_num'], index = df.index)
    df['expect_quota / expect_quota_mod - current_repay_amt_usd_min'] = pd.Series((df['expect_quota'] + 0.01) / (df['expect_quota_mod - current_repay_amt_usd_min']+0.01) ,index = df.index)
    df['300385 * current_bill_min_repay_amt_std'] = pd.Series(df['300385'] * df['current_bill_min_repay_amt_std'],index = df.index)
    df['gross_profit - 300385 * current_bill_min_repay_amt_std'] = pd.Series(np.array(normalize(df[['gross_profit']], axis = 0) - normalize(df[['300385 * current_bill_min_repay_amt_std']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * tag_6'] = pd.Series(df['education_max_info_num'] * df['tag_6'],index = df.index)
    df['local_hk + education_max_info_num * tag_6'] = pd.Series(np.array(normalize(df[['local_hk']], axis = 0) + normalize(df[['education_max_info_num * tag_6']], axis = 0)).ravel(), index = df.index)
    return df

colall = [u'age', u'sex', u'expect_quota', u'max_month_repay', u'occupation', u'education', u'marital_status', u'live_info', u'local_hk', u'money_function', u'company_type', u'salary', u'school_type', u'flow', u'gross_profit', u'business_type', u'business_year', u'personnel_num', u'pay_type', u'product_id', u'tm_encode', u'info_num', u'time_gap', u'tm_encode_max_max_month_repay_min', u'education_max_info_num', u'num', u'prior_period_bill_amt_max', u'prior_period_bill_amt_min', u'prior_period_bill_amt_mean', u'prior_period_bill_amt_std', u'prior_period_repay_amt_max', u'prior_period_repay_amt_min', u'prior_period_repay_amt_mean', u'prior_period_repay_amt_std', u'credit_lmt_amt_max', u'credit_lmt_amt_min', u'credit_lmt_amt_mean', u'credit_lmt_amt_std', u'curt_jifen_max', u'curt_jifen_min', u'curt_jifen_mean', u'curt_jifen_std', u'current_bill_bal_max', u'current_bill_bal_min', u'current_bill_bal_mean', u'current_bill_bal_std', u'current_bill_min_repay_amt_max', u'current_bill_min_repay_amt_min', u'current_bill_min_repay_amt_mean', u'current_bill_min_repay_amt_std', u'is_cheat_bill_max', u'is_cheat_bill_min', u'is_cheat_bill_mean', u'is_cheat_bill_std', u'cost_cnt_max', u'cost_cnt_min', u'cost_cnt_mean', u'cost_cnt_std', u'current_bill_amt_max', u'current_bill_amt_min', u'current_bill_amt_mean', u'current_bill_amt_std', u'adj_amt_max', u'adj_amt_min', u'adj_amt_mean', u'adj_amt_std', u'circle_interest_max', u'circle_interest_min', u'circle_interest_mean', u'circle_interest_std', u'prior_period_jifen_bal_max', u'prior_period_jifen_bal_min', u'prior_period_jifen_bal_mean', u'prior_period_jifen_bal_std', u'nadd_jifen_max', u'nadd_jifen_min', u'nadd_jifen_mean', u'nadd_jifen_std', u'current_adj_jifen_max', u'current_adj_jifen_min', u'current_adj_jifen_mean', u'current_adj_jifen_std', u'avlb_bal_usd_max', u'avlb_bal_usd_min', u'avlb_bal_usd_mean', u'avlb_bal_usd_std', u'avlb_bal_max', u'avlb_bal_min', u'avlb_bal_mean', u'avlb_bal_std', u'card_type_max', u'card_type_min', u'card_type_mean', u'card_type_std', u'pre_borrow_cash_amt_usd_max', u'pre_borrow_cash_amt_usd_min', u'pre_borrow_cash_amt_usd_mean', u'pre_borrow_cash_amt_usd_std', u'credit_lmt_amt_usd_max', u'credit_lmt_amt_usd_min', u'credit_lmt_amt_usd_mean', u'credit_lmt_amt_usd_std', u'pre_borrow_cash_amt_max', u'pre_borrow_cash_amt_min', u'pre_borrow_cash_amt_mean', u'pre_borrow_cash_amt_std', u'curr_max', u'curr_min', u'curr_mean', u'curr_std', u'repay_stat_max', u'repay_stat_min', u'repay_stat_mean', u'repay_stat_std', u'current_min_repay_amt_usd_max', u'current_min_repay_amt_usd_min', u'current_min_repay_amt_usd_mean', u'current_min_repay_amt_usd_std', u'current_repay_amt_usd_max', u'current_repay_amt_usd_min', u'current_repay_amt_usd_mean', u'current_repay_amt_usd_std', u'current_convert_jifen_max', u'current_convert_jifen_min', u'current_convert_jifen_mean', u'current_convert_jifen_std', u'current_award_jifen_max', u'current_award_jifen_min', u'current_award_jifen_mean', u'current_award_jifen_std', u'relation1_num', u'type0_num', u'type0_sum', u'type1_num', u'type1_sum', u'type2_num', u'type2_sum', u'type3_num', u'type3_sum', u'time_mean', u'time_min', u'time_max', u'time_std',  u'tag_num', u'tag_3', u'tag_4', u'tag_5', u'tag_6', u'tag_7', u'tag_8', u'tag_9', u'300028', u'300196', u'300301', u'300385', u'300469', u'300658', u'301036', u'301211', u'301687', u'300028_301687', u'300196_2']

selectcolori = ['relation1_num','education_max_info_num * adj_amt_min','nolabel','time_gap * pay_type','tm_encode_max', 'tag_num * 300196', 'tag_num / relation1_num', 'adj_amt_min / pay_type', 'education_max_info_num', 'time_gap', 'expect_quota_mod', 'prior_period_repay_amt_std', 'money_function_mean', 'current_award_jifen_max', 'prior_period_bill_amt_mean', 'salary_max', 'occupation_std', '301211', 'current_bill_amt_std', 'pre_borrow_cash_amt_usd_mean', '300196 * relation1_num', 'money_function_mean * current_award_jifen_max', 'tm_encode_max * max_month_repay', 'prior_period_bill_amt_mean * marital_status_min', 'max_month_repay * prior_period_bill_amt_mean', 'occupation_std / marital_status', '300385 * current_bill_min_repay_amt_std', 'tm_encode_max / curt_jifen_mean', 'education_max_info_num * tag_6', 'pay_type * nolabel', 'expect_quota_mod - current_repay_amt_usd_min', 'tm_encode_max / prior_period_repay_amt_std', 'expect_quota_mod - current_repay_amt_usd_min / salary_max', 'nolabel + prior_period_bill_amt_mean * marital_status_min', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std', 'money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std', 'current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max','time_gap * pay_type - occupation_std / marital_status', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num','prior_period_bill_amt_mean * marital_status_min + education_max_info_num * tag_6','expect_quota * money_function_mean * current_award_jifen_max','nolabel * tag_num / relation1_num','expect_quota / expect_quota_mod - current_repay_amt_usd_min']

def main(colall, selectcolori):
    train = pd.read_csv('train_new_label_2.csv')
    set = pd.read_csv('xgbimpute.csv')
    df = pd.merge(train, set, on = 'user_id', how = 'left', left_index = True)
    aucfix = 0.6927
    ColumnName = list(df.columns)
    ColumnName.remove('user_id')
    ColumnName.remove('lable')
    ColumnName.remove('lable2')

    col = selectcolori[:]
    print 'total round ' + str(len(col))
    for i in np.arange(len(col)):
        print str(col[i])
        for j in np.arange(len(col)-1,0,-1):
            selectcol =  selectcolori[:]
            df = newdataset(df, selectcol[:-11])
            inte = 4
            if j == len(col)-1:
                inte = 5 #add the feature itself
            for t in range(inte):
                x, name = addcolumns(col[i],col[j], df, df[selectcol].copy(), t);select(x,df,aucfix, name)
                #select(df[selectcol].copy(),df,aucfix);sys.exit() #for checking the auc score of selectcolori combination

if __name__ == "__main__":
    pool = Pool(8)
    colall = [u'age', u'sex', u'expect_quota', u'max_month_repay', u'occupation', u'education', u'marital_status', u'live_info', u'local_hk', u'money_function', u'company_type', u'salary', u'school_type', u'flow', u'gross_profit', u'business_type', u'business_year', u'personnel_num', u'pay_type', u'product_id', u'tm_encode', u'info_num', u'time_gap', u'tm_encode_max_max_month_repay_min', u'education_max_info_num', u'num', u'prior_period_bill_amt_max', u'prior_period_bill_amt_min', u'prior_period_bill_amt_mean', u'prior_period_bill_amt_std', u'prior_period_repay_amt_max', u'prior_period_repay_amt_min', u'prior_period_repay_amt_mean', u'prior_period_repay_amt_std', u'credit_lmt_amt_max', u'credit_lmt_amt_min', u'credit_lmt_amt_mean', u'credit_lmt_amt_std', u'curt_jifen_max', u'curt_jifen_min', u'curt_jifen_mean', u'curt_jifen_std', u'current_bill_bal_max', u'current_bill_bal_min', u'current_bill_bal_mean', u'current_bill_bal_std', u'current_bill_min_repay_amt_max', u'current_bill_min_repay_amt_min', u'current_bill_min_repay_amt_mean', u'current_bill_min_repay_amt_std', u'is_cheat_bill_max', u'is_cheat_bill_min', u'is_cheat_bill_mean', u'is_cheat_bill_std', u'cost_cnt_max', u'cost_cnt_min', u'cost_cnt_mean', u'cost_cnt_std', u'current_bill_amt_max', u'current_bill_amt_min', u'current_bill_amt_mean', u'current_bill_amt_std', u'adj_amt_max', u'adj_amt_min', u'adj_amt_mean', u'adj_amt_std', u'circle_interest_max', u'circle_interest_min', u'circle_interest_mean', u'circle_interest_std', u'prior_period_jifen_bal_max', u'prior_period_jifen_bal_min', u'prior_period_jifen_bal_mean', u'prior_period_jifen_bal_std', u'nadd_jifen_max', u'nadd_jifen_min', u'nadd_jifen_mean', u'nadd_jifen_std', u'current_adj_jifen_max', u'current_adj_jifen_min', u'current_adj_jifen_mean', u'current_adj_jifen_std', u'avlb_bal_usd_max', u'avlb_bal_usd_min', u'avlb_bal_usd_mean', u'avlb_bal_usd_std', u'avlb_bal_max', u'avlb_bal_min', u'avlb_bal_mean', u'avlb_bal_std', u'card_type_max', u'card_type_min', u'card_type_mean', u'card_type_std', u'pre_borrow_cash_amt_usd_max', u'pre_borrow_cash_amt_usd_min', u'pre_borrow_cash_amt_usd_mean', u'pre_borrow_cash_amt_usd_std', u'credit_lmt_amt_usd_max', u'credit_lmt_amt_usd_min', u'credit_lmt_amt_usd_mean', u'credit_lmt_amt_usd_std', u'pre_borrow_cash_amt_max', u'pre_borrow_cash_amt_min', u'pre_borrow_cash_amt_mean', u'pre_borrow_cash_amt_std', u'curr_max', u'curr_min', u'curr_mean', u'curr_std', u'repay_stat_max', u'repay_stat_min', u'repay_stat_mean', u'repay_stat_std', u'current_min_repay_amt_usd_max', u'current_min_repay_amt_usd_min', u'current_min_repay_amt_usd_mean', u'current_min_repay_amt_usd_std', u'current_repay_amt_usd_max', u'current_repay_amt_usd_min', u'current_repay_amt_usd_mean', u'current_repay_amt_usd_std', u'current_convert_jifen_max', u'current_convert_jifen_min', u'current_convert_jifen_mean', u'current_convert_jifen_std', u'current_award_jifen_max', u'current_award_jifen_min', u'current_award_jifen_mean', u'current_award_jifen_std', u'relation1_num', u'type0_num', u'type0_sum', u'type1_num', u'type1_sum', u'type2_num', u'type2_sum', u'type3_num', u'type3_sum', u'time_mean', u'time_min', u'time_max', u'time_std',  u'tag_num', u'tag_3', u'tag_4', u'tag_5', u'tag_6', u'tag_7', u'tag_8', u'tag_9', u'300028', u'300196', u'300301', u'300385', u'300469', u'300658', u'301036', u'301211', u'301687', u'300028_301687', u'300196_2']
    selectcolori = ['relation1_num','education_max_info_num * adj_amt_min','nolabel','time_gap * pay_type','tm_encode_max', 'tag_num * 300196', 'tag_num / relation1_num', 'adj_amt_min / pay_type', 'education_max_info_num', 'time_gap', 'expect_quota_mod', 'prior_period_repay_amt_std', 'money_function_mean', 'current_award_jifen_max', 'prior_period_bill_amt_mean', 'salary_max', 'occupation_std', '301211', 'current_bill_amt_std', 'pre_borrow_cash_amt_usd_mean', '300196 * relation1_num', 'money_function_mean * current_award_jifen_max', 'tm_encode_max * max_month_repay', 'prior_period_bill_amt_mean * marital_status_min', 'max_month_repay * prior_period_bill_amt_mean', 'occupation_std / marital_status', '300385 * current_bill_min_repay_amt_std', 'tm_encode_max / curt_jifen_mean', 'education_max_info_num * tag_6', 'pay_type * nolabel', 'expect_quota_mod - current_repay_amt_usd_min', 'tm_encode_max / prior_period_repay_amt_std', 'expect_quota_mod - current_repay_amt_usd_min / salary_max', 'nolabel + prior_period_bill_amt_mean * marital_status_min', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std', 'money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std', 'current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max','time_gap * pay_type - occupation_std / marital_status', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num','prior_period_bill_amt_mean * marital_status_min + education_max_info_num * tag_6','expect_quota * money_function_mean * current_award_jifen_max','nolabel * tag_num / relation1_num','expect_quota / expect_quota_mod - current_repay_amt_usd_min']
    main()