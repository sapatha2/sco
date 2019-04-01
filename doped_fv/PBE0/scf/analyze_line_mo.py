import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

#LINE SAMPLING
zz=0
E=[-9.2092669022287E+02]*2+\
  [-9.2092063265454E+02]*2+\
  [-9.2091969201655E+02]*2+\
  [-9.2091264054414E+02]*2
basestate=[0,0,1,1,2,2,3,3]
df=None
for name in ['FLP_up','FLP_dn','COL_up','COL_dn','COL2_up','COL2_dn','FM_up','FM_dn']:
  d=pd.read_pickle('pickles/'+name+'_line_gosling_g.pickle')
  d['energy']+=E[zz]
  d['energy']*=27.2114
  d['basestate']=basestate[zz]
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)
  zz+=1

'''
plt.plot(np.arange(df.shape[0]),df['energy'],'o')
plt.show()

print(max(df['energy'])-min(df['energy']))
exit(0)
'''

'''
#PAIRPLOT: On small data set, Nd + Nps + N4s is constant,
#there is a strong negative correlation with Nd + Nps and N4s,
#implying that we would want to trace cut Nd + Nps, hence not
#including states which have too much N4s occupation!
#sns.pairplot(df,vars=['energy','sigT','sigNps','sigNd','sigU'],hue='rem')
#sns.pairplot(df,vars=['energy','n'],hue='name',markers='.')
#plt.show()
#exit(0)

#APPLYING THE CUTOFFS:
ind=np.where(df['energy']==min(df['energy']))[0][0]

#Npi, Nz
cutoff=10
select=np.full(df.shape[0],True)
for field in ['sigNpp','sigNdpi','sigNpz','sigNdz']:
  n_gs=df.iloc[ind][field]
  select*=(df[field].values<(n_gs+cutoff))*\
         (df[field].values>(n_gs-cutoff))

#Energy
cutoff=1.0
select*=(df['energy'].values<=min(df['energy'])+cutoff)

#Plots
select_df=df.iloc[select]
sns.pairplot(select_df,vars=['energy','sigTd','sigT','sigNps','sigU','sigNpp'],hue='Sz')
#sns.pairplot(select_df,vars=['energy','n','sigN4s','n_tot'],hue='Sz')
plt.show()
exit(0)
'''

#REGRESSION
#ind=(np.abs(df['gsw']-1.0)<1e-10)+(np.abs(df['gsw'])<1e-10)
#select_df=df[ind]
#select_df['energy']-=min(select_df['energy'])
select_df=df
y=select_df['energy']

#plt.plot(select_df['sigN_0_0'],select_df['energy'],'o')
#plt.plot(select_df['sigU'],select_df['energy'],'o')
#plt.show()
#exit(0)

X=select_df[['sigN_1_1','sigN_2_2','sigN_3_3','sigN_4_4','sigN_5_5','sigN_6_6','sigN_7_7','sigU']]
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

#plt.plot(ols.predict(),y,'bo')
#for i in range(len(y)):
#  plt.plot([l_ols.values[i],u_ols.values[i]],[y.values[i],y.values[i]],'b-')
#plt.plot(y,y,'g--')
#plt.show()

