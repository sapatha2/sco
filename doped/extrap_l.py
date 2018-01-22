import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,0.05,0.1,0.15])
a=np.array([3.9278598630770247,3.9374878376296034,3.9472139916789573,3.95276179364695])
c=np.array([3.4338030848384617,3.421828204546191,3.4119182420528116,3.405746619540192])

ma,ba=np.polyfit(x,a,1)
mc,bc=np.polyfit(x,c,1)

a=np.insert(a,len(a),ma*0.25+ba)
c=np.insert(c,len(c),mc*0.25+bc)
x=np.insert(x,len(x),0.25)

plt.subplot(211)
plt.plot(x,a,'ob')
plt.plot(x,ma*x+ba,'-b')
plt.ylabel("a")
plt.xlabel("x")
plt.subplot(212)
plt.plot(x,c,'og')
plt.plot(x,mc*x+bc,'-g')
plt.ylabel("b")
plt.xlabel("x")
plt.show()
