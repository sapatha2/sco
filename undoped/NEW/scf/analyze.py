import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm

df=pd.read_pickle('scf_gosling.pickle')
df=df.iloc[:5]
#df.iloc[3:5]['Sz']=2
#df.iloc[5:7]['Sz']=3
#df['N']=df['sigNpp']+df['sigNd']+df['sigNps']#+df['sigN4s']+df['sigN2s'] #Last two terms get the leftover 25% of electron under excitation
#print(np.sqrt(df['N'].var()))
#df=df[df['Sz']==0]
sns.pairplot(df,vars=['energy','sigU','sigNps','sigT','sigNd'],hue='Sz',markers='o')
#X=df[['sigU','sigT','sigNps']]
#y=df['energy']
#X=sm.add_constant(X)
#ols=sm.OLS(y,X).fit()
#print(ols.summary())
plt.show()
