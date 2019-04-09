import pylab as P
import numpy as np
import sys
import math

def read_bands(filename):
  f=open(filename,'r')
  spl=f.readline().split()
  nkpt=int(spl[2])
  nband=int(spl[4])
  nspin=int(spl[6])
  spl=f.readline().split()
  npanel=int(spl[2])
  print(nkpt,nband,nspin,npanel)
  panels=[]
  for i in range(0,npanel+1):
    spl=f.readline().split()
    panels.append( (int(spl[1]),spl[2]) )
  data=[[],[]]
  for s in range(0,nspin):
    dummy=""
    while(dummy != "VIEW"):
      dummy=f.readline().split()[1]
    for k in range(1,nkpt):
      line=f.readline()
      #PYTHON2
      #data[s].append(map(float,line.split()))
      #PYTHON3
      data[s].append(list(map(float,line.split())))
  data[0]=np.array(data[0])
  data[1]=np.array(data[1])
  return data,panels

bands,panels=read_bands("BAND.DAT")
nband=bands[0].shape[1]
nkpt=bands[0].shape[0]
ax=P.gca()
for p in panels:
  ax.axvline(p[0],color='k')
  ax.annotate(p[1],xy=(p[0],0.0),ha='center',fontsize=9)
ax.axhline(0.0,color='gray',linestyle="--")
conv=27.2116
for b in range(1,80):
  if(bands[0][0,b]<0): print(b,"spin up")
  if(bands[1][0,b]<0): print(b,"spin down")
  ax.annotate(str(b), xy=(0, conv*bands[0][0,b]),fontsize=14, color='k')
  ax.annotate(str(b), xy=(120, conv*bands[1][-1,b]),fontsize=14, color='r')
  #ax.annotate(0, conv*bands[1][0,b], str(b), transform=ax.transAxes, fontsize=14,
  #        verticalalignment='top', color='r')
  ax.plot(range(1,nkpt+1),conv*bands[0][:,b],color='k')#,ls='--')
  ax.plot(range(1,nkpt+1),conv*bands[1][:,b],color='r')#,ls='--')
  '''
  if(b==66): ax.plot(range(1,nkpt+1),conv*bands[1][:,b],color='r')
  if(b==67): 
    ax.plot(range(1,nkpt+1),conv*bands[0][:,b],color='k')
    ax.plot(range(1,nkpt+1),conv*bands[1][:,b],color='r')
  if(b==68): ax.plot(range(1,nkpt+1),conv*bands[0][:,b],color='k')
  if(b==69): ax.plot(range(1,nkpt+1),conv*bands[1][:,b],color='r')
  '''
P.ylim([-10,10])
P.ylabel("Energy (eV) ")
if len(sys.argv) < 2: 
  P.show()
else:
  P.savefig(sys.argv[1])
