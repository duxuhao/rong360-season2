import pandas as pd
import numpy as np
from multiprocessing import Pool
import xgboost as xgb
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

def export(clf, df, feature):
    pre = pd.read_csv('test.txt')
    PredictionPredictor = df[feature]
    df['probability'] = np.round(clf.predict_proba(PredictionPredictor)[:,1], 5)
    result = pd.merge(pre, df[['user_id','probability']], on = 'user_id', how = 'left', left_index = None)
    result = result.drop_duplicates(cols = 'user_id')
    result = result.fillna(0.5)
    result.to_csv('../model1_answer.txt', index = None, encoding = 'utf-8')
    
def importance_check(clf, feature):
    imp = clf.feature_importances_
    a = [feature,imp]
    importance = pd.DataFrame(np.reshape(a,(2,-1)).T)
    importance.columns = ['feature','ratio']
    print importance.sort(['ratio'],ascending = False)

def add_multiple_combination(df):
    df['money_function_mean * current_award_jifen_max'] = pd.Series(df['money_function_mean'] * df['current_award_jifen_max'], index = df.index)
    df['expect_quota_mod - current_repay_amt_usd_min / salary_max']  = pd.Series((np.array(normalize(df[['expect_quota_mod']], axis = 0) - normalize(df[['current_repay_amt_usd_min']], axis = 0)).ravel()+0.01) / (df['salary_max']+0.01), index = df.index)
    df['nolabel + prior_period_bill_amt_mean * marital_status_min'] = pd.Series(np.array(normalize(df[['nolabel']], axis = 0) + normalize(df[['prior_period_bill_amt_mean * marital_status_min']],axis = 0)).ravel(), index = df.index)
    df['money_function_mean / tm_encode_max / prior_period_repay_amt_std'] = pd.Series((df['money_function_mean'] + 0.01) / (df['tm_encode_max / prior_period_repay_amt_std'] + 0.01), index = df.index)
    df['money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std'] = pd.Series(df['money_function_mean * current_award_jifen_max'] * df['tm_encode_max / prior_period_repay_amt_std'], index = df.index)
    df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series(np.array(normalize(df[['current_bill_amt_std']], axis = 0) - normalize(df[['expect_quota_mod - current_repay_amt_usd_min / salary_max']], axis = 0)).ravel(), index = df.index)
    df['time_gap * pay_type - occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['time_gap * pay_type']], axis = 0) - normalize(df[['occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(df['education_max_info_num'] * df['expect_quota_mod - current_repay_amt_usd_min'],index = df.index)
    df['nolabel * expect_quota_mod'] = pd.Series(df['nolabel'] * df['expect_quota_mod'], index = df.index)
    df['300385 * current_bill_min_repay_amt_std'] = pd.Series(df['300385'] * df['current_bill_min_repay_amt_std'], index = df.index)
    df['nolabel * expect_quota_mod - 300385 * current_bill_min_repay_amt_std'] = pd.Series(np.array(normalize(df[['300385 * current_bill_min_repay_amt_std']], axis = 0) - normalize(df[['300385 * current_bill_min_repay_amt_std']], axis = 0)).ravel(), index = df.index)
    df['nolabel + expect_quota_mod - current_repay_amt_usd_min / salary_max']  = pd.Series(np.array(normalize(df[['nolabel']], axis = 0) + normalize(df[['expect_quota_mod - current_repay_amt_usd_min / salary_max']], axis = 0)).ravel(), index = df.index)
    df['expect_quota_mod / tag_num / relation1_num'] = pd.Series((df['expect_quota_mod']+0.01)/(df['tag_num / relation1_num']+0.01),index = df.index)
    df['tag_num * 300196 - occupation_std'] = pd.Series(np.array(normalize(df[['tag_num * 300196']], axis = 0) - normalize(df[['occupation_std']], axis = 0)).ravel(), index = df.index)
    df['current_award_jifen_max * expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(np.array(normalize(df[['expect_quota_mod']], axis = 0) - normalize(df[['current_repay_amt_usd_min']], axis = 0)).ravel() * df['current_award_jifen_max'], index = df.index)
    df['expect_quota_mod - current_repay_amt_usd_min + salary_max'] = pd.Series(np.array(normalize(df[['expect_quota_mod - current_repay_amt_usd_min']], axis = 0) + normalize(df[['salary_max']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * tag_6 * tm_encode_max * max_month_repay'] = pd.Series(df['education_max_info_num'] * df['tag_6'] * df['tm_encode_max'] * df['max_month_repay'], index = df.index)
    df['tm_encode_max / time_gap * pay_type'] = pd.Series((df['tm_encode_max']+0.01) / (df['time_gap * pay_type']+0.01),index = df.index)
    df['tag_num * 300196 / current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series((df['tag_num * 300196']+0.01) / (df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max']+0.01), index = df.index)
    df['time_gap * pay_type - occupation_std / marital_status * 301211'] = pd.Series(df['time_gap * pay_type - occupation_std / marital_status'] * df['301211'], index = df.index)
    df['tm_encode_max / time_gap * pay_type + occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['tm_encode_max / time_gap * pay_type']], axis = 0) + normalize(df[['occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * adj_amt_min - tm_encode_max / time_gap * pay_type + occupation_std / marital_status'] = pd.Series(np.array(normalize(df[['education_max_info_num * adj_amt_min']], axis = 0) - normalize(df[['tm_encode_max / time_gap * pay_type + occupation_std / marital_status']], axis = 0)).ravel(), index = df.index)
    df['time_gap * pay_type - occupation_std / marital_status * 301211 * tm_encode_max / prior_period_repay_amt_std'] = pd.Series(df['time_gap * pay_type - occupation_std / marital_status * 301211'] * df['tm_encode_max / prior_period_repay_amt_std'], index = df.index)
    df['expect_quota_mod - current_repay_amt_usd_min * relation1_num'] = pd.Series(df['expect_quota_mod - current_repay_amt_usd_min'] * df['relation1_num'], index = df.index)
    df['money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num'] = pd.Series(np.array(normalize(df[['money_function_mean / tm_encode_max / prior_period_repay_amt_std']], axis = 0) + normalize(df[['relation1_num']], axis = 0)).ravel(), index = df.index)
    df['tm_encode_max / prior_period_repay_amt_std - time_gap'] = pd.Series(np.array(normalize(df[['tm_encode_max / prior_period_repay_amt_std']], axis = 0) - normalize(df[['time_gap']], axis = 0)).ravel(), index = df.index)
    df['money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std - expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(np.array(normalize(df[['money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std']], axis = 0) - normalize(df[['expect_quota_mod - current_repay_amt_usd_min']], axis = 0)).ravel(), index = df.index)
    df['tag_num / relation1_num * time_gap'] = pd.Series(df['tag_num / relation1_num'] * df['time_gap'], index = df.index)
    df['relation1_num + current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series(np.array(normalize(df[['relation1_num']], axis = 0) + normalize(df[['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max']], axis = 0)).ravel(), index = df.index)
    df['300385 * current_bill_min_repay_amt_std / current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series((df['300385 * current_bill_min_repay_amt_std']+0.01) / (df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max']+0.01), index = df.index)
    df['salary_max / money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num'] = pd.Series((df['salary_max']+0.01) / (df['money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num']+0.01), index = df.index)
    df['prior_period_bill_amt_mean * marital_status_min + education_max_info_num * tag_6'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean * marital_status_min']], axis = 0) + normalize(df[['education_max_info_num * tag_6']], axis = 0)).ravel(), index = df.index)
    df['pre_borrow_cash_amt_usd_mean * expect_quota_mod - current_repay_amt_usd_min'] = pd.Series(df['pre_borrow_cash_amt_usd_mean'] * df['expect_quota_mod - current_repay_amt_usd_min'], index = df.index)
    df['prior_period_bill_amt_mean + time_gap * pay_type'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean']], axis = 0) + normalize(df[['time_gap * pay_type']], axis = 0)).ravel(), index = df.index)
    df['current_award_jifen_max * current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'] = pd.Series(df['current_award_jifen_max'] * df['current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max'], index = df.index)
    df['prior_period_bill_amt_mean + time_gap * pay_type'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean']], axis = 0) + normalize(df[['time_gap * pay_type']], axis = 0)).ravel(), index = df.index)
    df['prior_period_bill_amt_mean + education_max_info_num * tag_6'] = pd.Series(np.array(normalize(df[['prior_period_bill_amt_mean']], axis = 0) - normalize(df[['time_gap * pay_type']], axis = 0)).ravel(), index = df.index)
    df['relation1_num + money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num'] = pd.Series(np.array(normalize(df[['relation1_num']], axis = 0) + normalize(df[['money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num']], axis = 0)).ravel(), index = df.index)
    df['education_max_info_num * adj_amt_min * max_month_repay * prior_period_bill_amt_mean'] = pd.Series(df['education_max_info_num * adj_amt_min'] * df['max_month_repay * prior_period_bill_amt_mean'], index = df.index)
    df['expect_quota * money_function_mean * current_award_jifen_max'] = pd.Series(df['expect_quota'] * df['money_function_mean * current_award_jifen_max'],index = df.index)
    df['nolabel * tag_num / relation1_num'] = pd.Series(df['nolabel'] * df['tag_num / relation1_num'], index = df.index)
    df['expect_quota / expect_quota_mod - current_repay_amt_usd_min'] = pd.Series((df['expect_quota'] + 0.01) / (df['expect_quota_mod - current_repay_amt_usd_min']+0.01) ,index = df.index)
    df['local_hk + education_max_info_num * tag_6'] = pd.Series(np.array(normalize(df[['local_hk']], axis = 0) + normalize(df[['education_max_info_num * tag_6']], axis = 0)).ravel(), index = df.index)
    df['gross_profit - 300385 * current_bill_min_repay_amt_std'] = pd.Series(np.array(normalize(df[['gross_profit']], axis = 0) - normalize(df[['300385 * current_bill_min_repay_amt_std']], axis = 0)).ravel(), index = df.index)
    return df


def main():
    train = pd.read_csv('train_new_label_2.csv')
    test = pd.read_csv('test_new_label_2.csv')
    set = pd.read_csv('xgbimpute.csv')
    #best feature combination
    good = ['relation1_num','education_max_info_num * adj_amt_min','nolabel','time_gap * pay_type','tm_encode_max', 'tag_num * 300196', 'tag_num / relation1_num', 'adj_amt_min / pay_type', 'education_max_info_num', 'time_gap',  'prior_period_repay_amt_std',  'current_award_jifen_max', 'prior_period_bill_amt_mean',  'occupation_std', '301211', 'current_bill_amt_std', 'pre_borrow_cash_amt_usd_mean', '300196 * relation1_num', 'tm_encode_max * max_month_repay', 'prior_period_bill_amt_mean * marital_status_min', 'max_month_repay * prior_period_bill_amt_mean', 'occupation_std / marital_status', '300385 * current_bill_min_repay_amt_std', 'tm_encode_max / curt_jifen_mean', 'education_max_info_num * tag_6', 'pay_type * nolabel', 'expect_quota_mod - current_repay_amt_usd_min', 'tm_encode_max / prior_period_repay_amt_std', 'expect_quota_mod - current_repay_amt_usd_min / salary_max', 'nolabel + prior_period_bill_amt_mean * marital_status_min', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std', 'money_function_mean * current_award_jifen_max * tm_encode_max / prior_period_repay_amt_std', 'current_bill_amt_std - expect_quota_mod - current_repay_amt_usd_min / salary_max','time_gap * pay_type - occupation_std / marital_status', 'money_function_mean / tm_encode_max / prior_period_repay_amt_std + relation1_num','prior_period_bill_amt_mean * marital_status_min + education_max_info_num * tag_6','expect_quota * money_function_mean * current_award_jifen_max','nolabel * tag_num / relation1_num','expect_quota / expect_quota_mod - current_repay_amt_usd_min']# 6198
    multicombine = 11 #the length of the multi cross data
    #multi cross data prepare
    df = pd.merge(train, set, on = 'user_id', how = 'left', left_index = True)
    df = newdataset(df, good[:-multicombine])
    df = add_multiple_combination(df)
    dfpredict = pd.merge(test, set, on = 'user_id', how = 'left', left_index = True)
    dfpredict = dfpredict.fillna(-100)
    dfpredict = newdataset(dfpredict, good[:-multicombine])
    dfpredict = add_multiple_combination(dfpredict)
    #train
    clf1  = xgb.XGBClassifier(n_estimators=125, max_depth = 3, learning_rate = 0.05)
    clf1.fit(df[good],df.lable)
    #check the importance
    importance_check(clf1, good)
    #predict
    export(clf1, dfpredict, good)

if __name__ == "__main__":
    pool = Pool(8)
    main()