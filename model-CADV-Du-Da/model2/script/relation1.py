# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import mode
#开始
relation1=pd.read_csv(r'D:\360datamining\data\firstdata\data\relation1.csv')
user=pd.read_csv(r'D:\360datamining\data\firstdata\data\user_info_tf2.csv')
user_list=pd.read_csv(r'D:\360datamining\data\firstdata\data\result\userlist.csv')
relationtemp1=pd.DataFrame(columns=['user_id','relation1_num'])
relationtemp1.user_id=relation1.groupby('user1_id').count().index
relationtemp1.relation1_num=relation1.groupby('user1_id').count().values
relation=pd.merge(relationtemp1,user_list,on='user_id',how='inner')
concat=pd.merge(user,relation,on='user_id',how='left')
# plt.plot(concat.index,concat.relation1_num)
# plt.show()
##############填充缺失值############
fill_mode=lambda g: g.fillna(mode(g).mode[0])
concat.fillna(concat.median(),inplace=True)
print concat.head(100)
concat.to_csv(r'D:\360datamining\data\firstdata\data\user_add_rls1.csv')