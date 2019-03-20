import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

#LINE SAMPLING
spin_df=pd.read_pickle('sd_big_gosling.pickle')  #Add on COL,FLP,FM
spin_df['rem']=0
spin_df['add']=0
df=pd.read_pickle('line_small_gosling.pickle')
df['Sz']=0
df['energy']+=min(spin_df['energy'])
df=pd.concat((df,spin_df.iloc[[-1,-2,-3]]),axis=0)
df['n']=df['sigNd']+df['sigNps']
df['n_tot']=df['sigNd']+df['sigNps']+df['sigN4s']

#PAIRPLOT: On small data set, Nd + Nps + N4s is constant,
#there is a strong negative correlation with Nd + Nps and N4s,
#implying that we would want to trace cut Nd + Nps, hence not
#including states which have too much N4s occupation!
#sns.pairplot(df,vars=['energy','sigT','sigNps','sigNd','sigU'],hue='rem')
#sns.pairplot(df,vars=['energy','n','sigN4s','n_tot'],hue='rem')
#plt.show()

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

