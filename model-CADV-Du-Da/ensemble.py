import pandas as pd

try:
    df1 = pd.read_csv('model1_answer.txt')
except:
    print 'Somthing wrong. Maybe due to the version of sklearn or xgboost. ',
    print 'please check the error message above'
    df1 = pd.read_csv('model1_answer_before.txt')

try:
    df2 = pd.read_csv('model2_answer.txt')
except:
    print 'Somthing wrong. Maybe due to the version of sklearn or xgboost. ',
    print 'please check the error message above'
    df2 = pd.read_csv('model2_answer_before.txt')

df_final = df1.copy()
df_final.probability = 0.535 * df1.probability + 0.465 * df2.probability
df_final.to_csv('CADV-Du-Da.txt', index = None)