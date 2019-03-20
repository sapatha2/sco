import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

#LINE SAMPLING
zz=0
E=[-9.2123749620223E+02,
   -9.2122638412999E+02,
   -9.2122636552756E+02,
   -9.2122636552756E+02,
   -9.2121137910381E+02,
   -9.2121137910381E+02]
df=None
for name in ['chk','col','flp_up','flp_dn','fm_up','fm_dn']:
  d=pd.read_pickle(name+'_line_gosling.pickle')
  d['name']=name.split("_")[0]
  d['energy']+=E[zz]
  d['n']=d['sigNd']+d['sigNps']
  print(d[d['energy']==min(d['energy'])][['n','name']].iloc[0])
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)
  zz+=1

df=pd.read_pickle('sd_big_gosling.pickle').iloc[[0,-1,-2,-3]]
df['name']=df['Sz']

df['n']=df['sigNd']+df['sigNps']
df['n_tot']=df['sigNd']+df['sigNps']+df['sigN4s']
print(df[['n','Sz']])

#PAIRPLOT: On small data set, Nd + Nps + N4s is constant,
#there is a strong negative correlation with Nd + Nps and N4s,
#implying that we would want to trace cut Nd + Nps, hence not
#including states which have too much N4s occupation!
#sns.pairplot(df,vars=['energy','sigT','sigNps','sigNd','sigU'],hue='rem')
#sns.pairplot(df,vars=['energy','n'],hue='name',markers='.')
#plt.show()
exit(0)

#APPLYING THE CUTOFF:
cutoff=0.05
ind=np.where(df['energy']==min(df['energy']))[0][0]
n_gs=df.iloc[ind]['n']
select=(df['n'].values<(n_gs+cutoff))*\
       (df['n'].values>(n_gs-cutoff))
select_df=df.iloc[select]
sns.pairplot(select_df,vars=['energy','sigT','sigNps','sigNd','sigU'],hue='Sz')
#sns.pairplot(select_df,vars=['energy','n','sigN4s','n_tot'],hue='rem')
plt.show()

#REGRESSION
'''
y=select_df['energy']
X=select_df[['sigT','sigU']]  #Do t and U have to be divided by stuff because of the super cell?
X=sm.add_constant(X)
beta=2
weights=np.exp(-beta*(y-min(y)))
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

g = sns.FacetGrid(select_df,hue='add')
g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)
'''

#plt.plot(ols.predict(),y,'bo')
#for i in range(len(y)):
#  plt.plot([l_ols.values[i],u_ols.values[i]],[y.values[i],y.values[i]],'b-')
#plt.plot(y,y,'g--')
#plt.show()

