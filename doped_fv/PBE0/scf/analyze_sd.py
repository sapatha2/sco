import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

df=pd.read_pickle('pickles/full_sd_gosling.pickle')

#TRACE CHECK
df['tr']=df['Nd']+df['Ns']+df['Np']
df['tr_group']=0
df['tr_group'][df['tr']<=20.7]=-1
df['tr_group'][df['tr']>=21]=1

df['tr_unp']=df['N0']+df['N1']+df['N2']+df['N3']+df['N4']+df['N5']+df['N6']+df['N7']

#"BAD" EXCITATIONS
#for g in [-1,0,1]:
#  ind=np.where(df['tr_group'].values==g)[0]
#  print(ind)
#exit(0)

#VARIANCES
var=df.var()
ind=np.argsort(var.values)
print(var.iloc[ind])
#exit(0)

#PAIRPLOTS
sns.pairplot(df,vars=['energy','tr','tr_unp'],hue='tr_group')
#sns.pairplot(df,vars=['energy','tr','Ndz2','Ndz','Ndpi','Npz','Npp','N2s'],hue='tr_group')
#sns.pairplot(df,vars=['energy','Np','Ns','sigTdp','sigTps','sigTds','sigUd','sigUp','sigUs'],hue='basestate')
#plt.savefig('HubbardPPIAO_full.pdf',bbox_inches='tight')
plt.show()
exit(0)

'''
#REGRESSION
select_df=df
y=select_df['energy']
X=select_df[['Ns','Np','sigTdp','sigTps','sigTds','sigUd']] 
#X=select_df[['Np','Ns','sigTdp','sigTds','sigTps','sigUd','sigJd']]  #Do t and U have to be divided by stuff because of the super cell?
X=sm.add_constant(X)
beta=0.
weights=np.exp(-beta*(y-min(y)))
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

g = sns.FacetGrid(select_df,hue='tr_group')
g.map(plt.errorbar, "pred", "energy", "energy_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)
'''
