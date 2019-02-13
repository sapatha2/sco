import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

df=pd.read_pickle('scf_gosling.pickle')
#df['N']=df['sigNpp']+df['sigNd']+df['sigNps']#+df['sigN4s']+df['sigN2s'] #Last two terms get the leftover 25% of electron under excitation
#print(np.sqrt(df['N'].var()))
sns.pairplot(df,vars=['sigNps','sigNd','sigT','sigU'],hue='Sz')
plt.show()
