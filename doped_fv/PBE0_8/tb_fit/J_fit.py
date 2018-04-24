import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

#Measured energies
EPBE0=np.array([-1.8418601953776E+03,-1.8418566128030E+03,-1.8418522579094E+03,-1.8418450389802E+03,-1.8418284707358E+03])
EPBE0-=EPBE0[0]
EPBE0*=27.2

sigJ=np.array([0,0,0,-4,0])
