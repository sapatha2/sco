import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA
from functools import reduce 

df=pd.read_pickle('pickles/sd_gosling_mo.pickle')#.iloc[[0,1,6,7,12,13,18,19,20]]
var=df.var()
ind=np.argsort(var.values)
#print(var.iloc[ind])
#exit(0)
#plt.matshow(df[['energy','sigN_0','sigN_1','sigN_2','sigN_3','sigN_4','sigN_5','sigN_6','sigN_7','sigU','sigJ']].corr(),vmax=1,vmin=-1,cmap=plt.cm.bwr)
#plt.show()
#exit(0)

'''
#REGRESSION
select_df=df
y=select_df['energy']
#plt.plot(df['sigN_0_0'],df['energy'],'o')
#plt.plot(select_df['sigU'],select_df['energy'],'o')
#plt.show()
#exit(0)

#X=select_df[['sigN_1_1','sigN_2_2','sigN_3_3','sigN_4_4','sigN_5_5','sigN_6_6','sigN_7_7',
#'sigN_0_1','sigN_0_2','sigN_1_2','sigN_1_4','sigN_5_6','sigN_5_7','sigN_6_7','sigU']]
#X=select_df[['sigN_1_1','sigN_2_2','sigN_3_3','sigN_4_4','sigN_5_5','sigN_6_6','sigN_7_7','sigN_6_7','sigU']]
#X=select_df[['sigN_1_1','sigN_2_2','sigN_3_3','sigN_4_4','sigN_5_5','sigN_6_6','sigN_7_7','sigU']]
#X=select_df.drop(columns=['energy','basestate','sigJ'])  #Do t and U have to be divided by stuff because of the super cell?
#X=select_df[['sigN_1_1','sigN_2_2','sigN_3_3','sigN_4_4','sigU',
#'sigN_0_1','sigN_0_2','sigN_0_3','sigN_0_4']]
#'sigN_1_2','sigN_1_3','sigN_1_4']]
#'sigN_2_3','sigN_2_4']]
#'sigN_3_4']]
#'sigJ']]
X=select_df[['sigU','sigJ','sigN_55_55','sigN_65_65','sigN_66_66','sigN_67_67','sigN_68_68','sigN_69_69','sigN_70_70']]
X=sm.add_constant(X)
beta=0
weights=np.exp(-beta*(y-min(y)))
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

g = sns.FacetGrid(select_df,hue='basestate')
g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)
'''

#ROTATE INTO IAO BASIS
H1=np.diag([-19.5044,-15.0972,-14.5102,-4.5184,-9.0239,-1.1715,-1.7981,0])
mo_coeff=np.load('pickles/UNPOL_mo_coeff_g.pickle')[0][:,[55,65,66,67,68,69,70,71]]
a=np.load('pickles/iao_g.pickle')[:,[13,23,33,43,5,15,25,35,45,49,53,57,62,66,70,74]]
s=np.load('pickles/UNPOL_s_g.pickle')

mo_to_iao=reduce(np.dot,(mo_coeff.T,s,a))
e1=reduce(np.dot,(mo_to_iao.T,H1,mo_to_iao))
plt.matshow(e1,vmin=-5,vmax=5,cmap=plt.cm.bwr)
plt.show()
