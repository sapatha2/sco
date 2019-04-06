import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA
from functools import reduce 
from sklearn.linear_model import OrthogonalMatchingPursuit

#########################################################################################
#REDUCING
df=pd.read_pickle('pickles/sd_gosling_mo.pickle')
#df=df.iloc[[0,1,2,6,7,8,12,13,14,18,16,20,24,25,26,30,31,32,36,37,41,42,43,44]] #No 70,71,72
#df=df.iloc[[0,12,24,36]] #Spin states only

df['sigN_65_65']+=df['sigN_66_66']
df['sigN_69_69']+=df['sigN_70_70']
df=df.drop(columns=['sigN_66_66','sigN_70_70'])

df['sigT_0_1']+=df['sigT_0_2']
df['sigT_0_5']+=df['sigT_0_6']
df['sigT_1_3']+=df['sigT_2_3']
df['sigT_1_4']+=df['sigT_2_4']
df['sigT_1_5']+=df['sigT_1_6']+df['sigT_2_5']+df['sigT_2_6']
df['sigT_1_7']+=df['sigT_2_7']
df['sigT_3_5']+=df['sigT_3_6']
df['sigT_4_5']+=df['sigT_4_6']
df['sigT_5_7']+=df['sigT_6_7']
df=df.drop(columns=['sigT_1_2','sigT_5_6','sigT_0_2','sigT_0_6','sigT_2_3',
'sigT_2_4','sigT_1_6','sigT_2_5','sigT_2_6','sigT_2_7','sigT_3_6','sigT_4_6','sigT_6_7'])

'''
#for i in [0,3,4,7]:
#  df['sigT_'+str(i)+'_1']+=df['sigT_'+str(i)+'_2']
#  df['sigT_'+str(i)+'_5']+=df['sigT_'+str(i)+'_6']
#  df=df.drop(columns=['sigT_'+str(i)+'_2','sigT_'+str(i)+'_6'])

##########################################################################################
#ANALYSIS
select=['sigN_55_55','sigN_65_65','sigN_67_67','sigN_68_68','sigN_69_69','sigN_71_71']
for x in list(df):
  if('sigT' in x): select+=[x]
#sns.pairplot(select_df,vars=['energy']+select,hue='basestate')
#plt.show()
#plt.savefig('pp_base.pdf',bbox_inches='tight')
#exit(0)
select_df=df[['energy','sigU','basestate']+select]

X=select_df.drop(columns=['energy','basestate','sigN_65_65'])
y=select_df[['energy']]
for i in range(1,len(list(X))+1):
  print(i,'---------------------------------')
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X, y)
  coef = omp.coef_
  idx_r, = coef.nonzero()
  idy_r  = np.where(coef==0)[0]
  X_sub=X[np.array(list(X))[idx_r]]
  X_sub=sm.add_constant(X_sub)
  
  ols=sm.OLS(y,X_sub).fit()
  #plt.stem(np.ones(len(list(X))), coef)
  #plt.show()
  if(i==9):
    plt.plot(ols.predict(),y,'o')
    plt.show()
'''  
'''
  print(ols.summary())
  plt.subplot(211)
  plt.plot(i,ols.rsquared,'bo')
  plt.xlabel('# Model parameters')
  plt.ylabel('R2')
  plt.subplot(212)
  plt.plot(i,ols.condition_number,'go')
  plt.xlabel('# Model parameters')
  plt.ylabel('Cond #')
plt.show()
exit(0)
'''

select_df=df
#X=select_df.drop(columns=['energy','basestate','sigU','sigN_68_68'])
y=select_df['energy']
X=select_df[['sigN_55_55','sigN_67_67','sigN_68_68','sigN_69_69','sigN_71_71','sigU']]
X=sm.add_constant(X)
beta=0
weights=np.exp(-beta*y)
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

sns.pairplot(select_df,vars=['resid','sigU'],hue='basestate')
plt.show()

g = sns.FacetGrid(select_df,hue='basestate')
g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)

#ROTATE INTO IAO BASIS
H1=np.diag([-19.5044,-15.0972,-14.5102,-4.5184,-9.0239,-1.1715,-1.7981,0])
mo_coeff=np.load('pickles/UNPOL_mo_coeff_g.pickle')[0][:,[55,65,66,67,68,69,70,71]]
a=np.load('pickles/iao_g.pickle')[:,[13,23,33,43,5,15,25,35,45,49,53,57,62,66,70,74]]
s=np.load('pickles/UNPOL_s_g.pickle')

mo_to_iao=reduce(np.dot,(mo_coeff.T,s,a))
e1=reduce(np.dot,(mo_to_iao.T,H1,mo_to_iao))
plt.matshow(e1,vmin=-5,vmax=5,cmap=plt.cm.bwr)
plt.show()
