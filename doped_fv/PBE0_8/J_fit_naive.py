import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

'''
#All
sigJ=np.array([-4,-4,0,-2,-2,-2,0,0,0,0,0,0,4,4])
name=np.array(['CHK0','CHK2','COL0','FLP0','FLP2','FLP4','BCOL0','BCOL2','BLK0','ACHN2','ACHN4','ACHN6','FM8','FM10'])
E_PBE0=np.array([-1.8418450389802E+03,-1.8418562836080E+03,-1.8418601953776E+03,
-1.8418509420788E+03,-1.8418566128030E+03,-1.8418431902964E+03,
-1.8418284707358E+03,-1.8418497382121E+03,-1.8418522579094E+03,
-1.8418562836088E+03,-1.8418248027241E+03,-1.8418309653762E+03,
-1.8417482260516E+03,-1.8416505318649E+03])

ind=np.argsort(E_PBE0)
E_sort=E_PBE0[ind]
name_sort=name[ind]
plt.plot((E_sort-E_sort[0])*27.2,'o')
plt.xticks(np.arange(0,14),name_sort)
plt.show()
'''
name=np.array(['CHK0','COL0','FLP2','BCOL0','BLK0','ACHN4','FM8'])
E_PBE0=np.array([-1.8418450389802E+03,-1.8418601953776E+03,
-1.8418566128030E+03,-1.8418284707358E+03,-1.8418522579094E+03,
-1.8418248027241E+03,-1.8417482260516E+03])

ind=np.argsort(E_PBE0)
E_sort=E_PBE0[ind]
name_sort=name[ind]
plt.plot((E_sort-E_sort[0])*27.2,'o')
plt.xticks(np.arange(0,14),name_sort)
plt.show()
