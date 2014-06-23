# -*- coding: utf-8 -*-
"""
Created on Sun Jun 08 16:50:38 2014

@author: dashamstyr
"""

import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import process_tools as ptools
import lidar_tools as ltools
from scipy import signal
import MPLtools as mtools


def maxmin(arrayin,widths,f):
    #find all local maxima or minina from each row of an array and return array
    #of index values for each max or min
    
    arrayout=[]
    for n in range(len(arrayin)):
        temp=signal.argrelextrema(arrayin[n],f)[0]        
        for t in temp:
            arrayout.append((widths[n],t))
        
    return arrayout

def layer_filter(prof,maxiloc,miniloc,sigma0=[],thresh=3):
    """
    takes in an array of minima and maxima from CWT for a profile and calculates
    layer edges and peaks while filtering out peaks for which the delta from 
    edge to peak is less than some multiple of the shot noise from background and
    dark current
    
    inputs:
    prof - a pandas series represeting a single profile of lidar returns with altitude
    maxix - a list of maximum index values from the CWT results at a given wavelet width
            represent the peaks of a given layer
    minix - a list of minimum index values from the CWT results at a given wavelet width
            represnt the edges of a given layer
    sigma0 - baeline noise level for the profile, if empty it is calculated
    thresh - difference between peak and edge of a layer must exvceeed this multiple of sigma0 to be counted
    
    """
    #step 1:calculate noise floor, if not defined

    if not sigma0:
        SNRprof=ptools.calc_SNR(prof)
        sigma0=np.mean(SNRprof.values[-100:])
    #Step 2: Calculate profile values at each edge and peak
    layers=[]
    for peakloc in maxiloc:
        try:
            edge_below=[v for v in miniloc if v<peakloc][-1]
        except IndexError:
            continue
        edge_above_list=[v for v in miniloc if v>peakloc]
        if not edge_above_list:
            continue
        #Step 3: Calculate delta signal between peak and lower edge (directly before)
        delta_lower=prof.iloc[peakloc]-prof.iloc[edge_below]
        
        #Step 4: Filter out false layers for which delta < thresh*signam0
        if delta_lower>thresh*sigma0:
            #try to find upper edge where delta_upper exceeds threshold
            for e in edge_above_list:
                delta_upper=prof.iloc[peakloc]-prof.iloc[e]
                if delta_upper>thresh*sigma0:
                    edge_above=e
                    #if upper edge is found, add indices of (lower,center,upper) to layers
                    layers.append((edge_below,peakloc,edge_above))
                    break
   #Step 5: Return list of tuples where each element is ()   
   return layers
            
    
    
    

#open a test case from an actual .mpl file

os.chdir('C:\Users\dashamstyr\Dropbox\Lidar Files\MPL Data\DATA\Whistler-0330\Processed')

mpltest = mtools.MPL()

mpltest.fromHDF('201403300000-201403302300_proc.h5')

testprof = mpltest.data[0].iloc[80,:]
depolprof = mpltest.data[0].iloc[101,:]
real_z=testprof.index.values

z = np.arange(100,15000,3)
z = z+0.0

wave = 532.0  #nm
lrat_p = 30.0
lrat_m = 8.0*np.pi/3.0

P_mol = ltools.molprof(z,wave)

z_layer = np.arange(1000,2000,5,dtype=np.float)

beta_layer = np.ones_like(z_layer)*5e-6

layer = pan.Series(data=beta_layer, index=z_layer)

P_1 = ltools.addlayer(P_mol,layer,lrat_p)

p_rangecor0 = pan.Series(P_mol.loc['vals'].values*(z**2),index=z)
p_norm0 = p_rangecor0/p_rangecor0.iloc[0]
p_rangecor1 = pan.Series(P_1.loc['vals'].values*(z**2) ,index=z)  
p_norm1 = p_rangecor1/p_rangecor1.iloc[0]

p_noisy = ptools.backandnoise(p_norm1)



wavelet=signal.ricker

widths=np.arange(2,51)

cwt_clean=signal.cwt(p_norm0,wavelet,widths)
cwt_noisy=signal.cwt(p_noisy,wavelet,widths)
cwt_real=signal.cwt(testprof,wavelet,widths)
cwt_depol=signal.cwt(depolprof,wavelet,widths)
#clean_peaks=signal.find_peaks_cwt(p_norm0,widths,wavelet)
#real_peaks=signal.find_peaks_cwt(testprof,widths,wavelet)

max_clean=maxmin(cwt_clean,widths,np.greater)
min_clean=maxmin(cwt_clean,widths,np.less)
max_real=maxmin(cwt_real,widths,np.greater)
min_real=maxmin(cwt_real,widths,np.less)
max_depol=maxmin(cwt_depol,widths,np.greater)
min_depol=maxmin(cwt_depol,widths,np.less)


minloc1=[i[1] for i in min_real if i[0]==2]
maxloc1=[i[1] for i in max_real if i[0]==2]

layers1=layer_filter(testprof,maxloc1,minloc1)

[baseiloc1,peakiloc1,topiloc1]=[list(l) for l in zip(*layers1)]

basevals1=testprof.iloc[baseiloc1]
baseix1=testprof.index[baseiloc1]
peakvals1=testprof.iloc[peakiloc1]
peakix1=testprof.index[peakiloc1]
topvals1=testprof.iloc[topiloc1]
topix1=testprof.index[topiloc1]

minloc2=[i[1] for i in min_depol if i[0]==2]
maxloc2=[i[1] for i in max_depol if i[0]==2]

layers2=layer_filter(depolprof,maxloc2,minloc2)

[baseiloc2,peakiloc2,topiloc2]=[list(l) for l in zip(*layers2)]

basevals2=depolprof.iloc[baseiloc2]
baseix2=depolprof.index[baseiloc2]
peakvals2=depolprof.iloc[peakiloc2]
peakix2=depolprof.index[peakiloc2]
topvals2=depolprof.iloc[topiloc2]
topix2=depolprof.index[topiloc2]

fig1=plt.figure(1)
ax1=plt.subplot2grid((3,4),(0,0),colspan=2)
ax1.plot(depolprof.index,depolprof.values)
ax1.plot(baseix2,basevals2,'ro',peakix2,peakvals2,'gv',topix2,topvals2,'bx')
ax2=plt.subplot2grid((3,4),(1,0),colspan=2)
ax2.imshow(cwt_depol,aspect=3)

ax3=plt.subplot2grid((3,4),(2,0),colspan=2)
ax3.plot(zip(*max_depol)[1],zip(*max_depol)[0],'k.',zip(*min_depol)[1],zip(*min_depol)[0],'r.')
ax3.invert_yaxis()
    
ax4=plt.subplot2grid((3,4),(0,2),colspan=2)
ax4.plot(testprof.index,testprof.values)
ax4.plot(baseix1,basevals1,'ro',peakix1,peakvals1,'gv',topix1,topvals1,'bx')
ax5=plt.subplot2grid((3,4),(1,2),colspan=2)
ax5.imshow(cwt_real,aspect=3)

ax6=plt.subplot2grid((3,4),(2,2),colspan=2)
ax6.plot(zip(*max_real)[1],zip(*max_real)[0],'k.',zip(*min_real)[1],zip(*min_real)[0],'r.')
ax6.invert_yaxis()    
fig1.canvas.draw()



del fig1
