import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
import xgboost as xgb
from multiprocessing import Pool

pool = Pool(8)
feature = [u'age', u'sex', u'expect_quota', u'max_month_repay', u'occupation', u'education', u'marital_status_min', u'live_info', u'local_hk', u'money_function_mean', u'company_type', u'salary_max', u'school_type', u'flow', u'gross_profit', u'business_type', u'business_year', u'personnel_num', u'pay_type', u'product_id', u'tm_encode', u'info_num', u'time_gap', u'tm_encode_max_max_month_repay_min', u'education_max_info_num', u'num', u'prior_period_bill_amt_max', u'prior_period_bill_amt_min', u'prior_period_bill_amt_mean', u'prior_period_bill_amt_std', u'prior_period_repay_amt_max', u'prior_period_repay_amt_min', u'prior_period_repay_amt_mean', u'prior_period_repay_amt_std', u'credit_lmt_amt_max', u'credit_lmt_amt_min', u'credit_lmt_amt_mean', u'credit_lmt_amt_std', u'curt_jifen_max', u'curt_jifen_min', u'curt_jifen_mean', u'curt_jifen_std', u'current_bill_bal_max', u'current_bill_bal_min', u'current_bill_bal_mean', u'current_bill_bal_std', u'current_bill_min_repay_amt_max', u'current_bill_min_repay_amt_min', u'current_bill_min_repay_amt_mean', u'current_bill_min_repay_amt_std', u'is_cheat_bill_max', u'is_cheat_bill_min', u'is_cheat_bill_mean', u'is_cheat_bill_std', u'cost_cnt_max', u'cost_cnt_min', u'cost_cnt_mean', u'cost_cnt_std', u'current_bill_amt_max', u'current_bill_amt_min', u'current_bill_amt_mean', u'current_bill_amt_std', u'adj_amt_max', u'adj_amt_min', u'adj_amt_mean', u'adj_amt_std', u'circle_interest_max', u'circle_interest_min', u'circle_interest_mean', u'circle_interest_std', u'prior_period_jifen_bal_max', u'prior_period_jifen_bal_min', u'prior_period_jifen_bal_mean', u'prior_period_jifen_bal_std', u'nadd_jifen_max', u'nadd_jifen_min', u'nadd_jifen_mean', u'nadd_jifen_std', u'current_adj_jifen_max', u'current_adj_jifen_min', u'current_adj_jifen_mean', u'current_adj_jifen_std', u'avlb_bal_usd_max', u'avlb_bal_usd_min', u'avlb_bal_usd_mean', u'avlb_bal_usd_std', u'avlb_bal_max', u'avlb_bal_min', u'avlb_bal_mean', u'avlb_bal_std', u'card_type_max', u'card_type_min', u'card_type_mean', u'card_type_std', u'pre_borrow_cash_amt_usd_max', u'pre_borrow_cash_amt_usd_min', u'pre_borrow_cash_amt_usd_mean', u'pre_borrow_cash_amt_usd_std', u'credit_lmt_amt_usd_max', u'credit_lmt_amt_usd_min', u'credit_lmt_amt_usd_mean', u'credit_lmt_amt_usd_std', u'pre_borrow_cash_amt_max', u'pre_borrow_cash_amt_min', u'pre_borrow_cash_amt_mean', u'pre_borrow_cash_amt_std', u'curr_max', u'curr_min', u'curr_mean', u'curr_std', u'repay_stat_max', u'repay_stat_min', u'repay_stat_mean', u'repay_stat_std', u'current_min_repay_amt_usd_max', u'current_min_repay_amt_usd_min', u'current_min_repay_amt_usd_mean', u'current_min_repay_amt_usd_std', u'current_repay_amt_usd_max', u'current_repay_amt_usd_min', u'current_repay_amt_usd_mean', u'current_repay_amt_usd_std', u'current_convert_jifen_max', u'current_convert_jifen_min', u'current_convert_jifen_mean', u'current_convert_jifen_std', u'current_award_jifen_max', u'current_award_jifen_min', u'current_award_jifen_mean', u'current_award_jifen_std', u'relation1_num', u'type0_num', u'type0_sum', u'type1_num', u'type1_sum', u'type2_num', u'type2_sum', u'type3_num', u'type3_sum', u'time_mean', u'time_min', u'time_max', u'time_std',  u'tag_num', u'tag_3', u'tag_4', u'tag_5', u'tag_6', u'tag_7', u'tag_8', u'tag_9', u'300028', u'300196', u'300301', u'300385', u'300469', u'300658', u'301036', u'301211', u'301687']#feature will be used

feature1 = [u'age', u'sex', u'expect_quota', u'max_month_repay', u'occupation', u'education', u'marital_status_min', u'live_info', u'local_hk', u'money_function_mean', u'company_type', u'salary_max', u'school_type', u'flow', u'gross_profit', u'business_type', u'business_year', u'personnel_num', u'pay_type', u'product_id', u'tm_encode', u'info_num', u'time_gap', u'tm_encode_max_max_month_repay_min', u'education_max_info_num'] #features from user_info

feature2 = [u'num', u'prior_period_bill_amt_max', u'prior_period_bill_amt_min', u'prior_period_bill_amt_mean', u'prior_period_bill_amt_std', u'prior_period_repay_amt_max', u'prior_period_repay_amt_min', u'prior_period_repay_amt_mean', u'prior_period_repay_amt_std', u'credit_lmt_amt_max', u'credit_lmt_amt_min', u'credit_lmt_amt_mean', u'credit_lmt_amt_std', u'curt_jifen_max', u'curt_jifen_min', u'curt_jifen_mean', u'curt_jifen_std', u'current_bill_bal_max', u'current_bill_bal_min', u'current_bill_bal_mean', u'current_bill_bal_std', u'current_bill_min_repay_amt_max', u'current_bill_min_repay_amt_min', u'current_bill_min_repay_amt_mean', u'current_bill_min_repay_amt_std', u'is_cheat_bill_max', u'is_cheat_bill_min', u'is_cheat_bill_mean', u'is_cheat_bill_std', u'cost_cnt_max', u'cost_cnt_min', u'cost_cnt_mean', u'cost_cnt_std', u'current_bill_amt_max', u'current_bill_amt_min', u'current_bill_amt_mean', u'current_bill_amt_std', u'adj_amt_max', u'adj_amt_min', u'adj_amt_mean', u'adj_amt_std', u'circle_interest_max', u'circle_interest_min', u'circle_interest_mean', u'circle_interest_std', u'prior_period_jifen_bal_max', u'prior_period_jifen_bal_min', u'prior_period_jifen_bal_mean', u'prior_period_jifen_bal_std', u'nadd_jifen_max', u'nadd_jifen_min', u'nadd_jifen_mean', u'nadd_jifen_std', u'current_adj_jifen_max', u'current_adj_jifen_min', u'current_adj_jifen_mean', u'current_adj_jifen_std', u'avlb_bal_usd_max', u'avlb_bal_usd_min', u'avlb_bal_usd_mean', u'avlb_bal_usd_std', u'avlb_bal_max', u'avlb_bal_min', u'avlb_bal_mean', u'avlb_bal_std', u'card_type_max', u'card_type_min', u'card_type_mean', u'card_type_std', u'pre_borrow_cash_amt_usd_max', u'pre_borrow_cash_amt_usd_min', u'pre_borrow_cash_amt_usd_mean', u'pre_borrow_cash_amt_usd_std', u'credit_lmt_amt_usd_max', u'credit_lmt_amt_usd_min', u'credit_lmt_amt_usd_mean', u'credit_lmt_amt_usd_std', u'pre_borrow_cash_amt_max', u'pre_borrow_cash_amt_min', u'pre_borrow_cash_amt_mean', u'pre_borrow_cash_amt_std', u'curr_max', u'curr_min', u'curr_mean', u'curr_std', u'repay_stat_max', u'repay_stat_min', u'repay_stat_mean', u'repay_stat_std', u'current_min_repay_amt_usd_max', u'current_min_repay_amt_usd_min', u'current_min_repay_amt_usd_mean', u'current_min_repay_amt_usd_std', u'current_repay_amt_usd_max', u'current_repay_amt_usd_min', u'current_repay_amt_usd_mean', u'current_repay_amt_usd_std', u'current_convert_jifen_max', u'current_convert_jifen_min', u'current_convert_jifen_mean', u'current_convert_jifen_std', u'current_award_jifen_max', u'current_award_jifen_min', u'current_award_jifen_mean', u'current_award_jifen_std'] #features from user_consumption

feature3 = [u'relation1_num'] #features from relation1

feature4 = [u'type0_num', u'type0_sum', u'type1_num', u'type1_sum', u'type2_num', u'type2_sum', u'type3_num', u'type3_sum', u'time_mean', u'time_min', u'time_max', u'time_std'] #feature from relation2

feature5 = [u'tag_num', u'tag_3', u'tag_4', u'tag_5', u'tag_6', u'tag_7', u'tag_8', u'tag_9', u'300028', u'300196', u'300301', u'300385', u'300469', u'300658', u'301036', u'301211', u'301687','300028_301687','300196_2'] #feature from tag



df = pd.read_csv('completetrain.csv')
train = df.ix[:,df.columns != 'lable']
df = pd.read_csv('completetest.csv')
df = df.drop_duplicates(cols = 'user_id')
test = df.ix[:,df.columns != 'probability']

df = pd.concat([train,test])
df.loc[df.relation1_num == 0,'relation1_num'] = -1
df = df.replace(-1,np.NaN)
#check the feature one by one, use the feature in other feature file to predictt the mission one. not using the features from the same file as it almost always miss together 
for label in feature:
    print label
    if label in feature1:
        usedfeature = feature2 +feature3 + feature4 + feature5
    elif label in feature2:
        usedfeature = feature1 +feature3 + feature4 + feature5
    elif label in feature3:
        usedfeature = feature2 +feature1 + feature4 + feature5
    elif label in feature4:
        usedfeature = feature2 +feature3 + feature1 + feature5
    elif label in feature5:
        usedfeature = feature2 +feature3 + feature4 + feature1

    new = df[~pd.isnull(df[label])]
    new = new.fillna(new.mean())
    X_train, X_test, y_train, y_test=train_test_split(new[usedfeature], new[label], test_size = 0.3, random_state = 200)
    if len(np.unique(new[label])) < 3:
        pass
        estimator = xgb.XGBClassifier(n_estimators=123, max_depth = 3, learning_rate = 0.05)
        estimator.fit(X_train, y_train)#,eval_metric = 'auc', eval_set=[(X_train, y_train), (X_test, y_test)])
        try:
            df.loc[pd.isnull(df[label]),label] = estimator.predict(df[pd.isnull(df[label])][usedfeature].as_matrix())
        except:
            df.loc[pd.isnull(df[label]),label] = -100
    else:
        estimator = xgb.XGBRegressor(n_estimators=123, max_depth = 3, learning_rate = 0.05)
        if estimator.score(X_test, y_test) < 0.2: #if the score higher than 0.2, use the algorithm to predict the missing value, if not, use -100 to fill the missing value
            df.loc[pd.isnull(df[label]),label] = -100
        else:
            try:
                df.loc[pd.isnull(df[label]),label] = estimator.predict(df[pd.isnull(df[label])][usedfeature].as_matrix())
            except:
                df.loc[pd.isnull(df[label]),label] = -100
    print estimator.score(X_test, y_test)

df.to_csv('xgbimpute.csv',index = None)
