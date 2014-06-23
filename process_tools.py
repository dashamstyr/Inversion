def backandnoise(P_in,background = 0,stdev = [],inplace=True):
    """
    Adds gaussian random noise and background signal to any profile
    
    Inputs:
    P_in = a pandas series with altitude index
    background = a float defining the backgorund signal to be applied(defaults to 0)
    stdev = the standard deviation of the gaussian noise component
    
    if no standard deviation is defined, the noise added is standard
    shot noise - poisson distribution approximated by a gaussian with
    std = sqrt(signal)
    
    Outputs:
    P_out = a copy of P_in with background signal and noise applied
    """
    
    import numpy as np
    import random
    from copy import deepcopy
    
    if inplace:
        P_out = P_in
    else:
        P_out=deepcopy(P_in)
    
    if stdev:
        P_out.values[:] = [v+random.gauss(background,stdev) for v in P_out.values]
    else:
        P_out.values[:] = [v+random.gauss(background,np.sqrt(v+background)) for v in P_out.values]
    return P_out

def background_subtract(P_in,back_avg=[],z_min=6000):
    #subtracts background from signal, without background
    #takes advantage of inverse square law to calculate background signal for
    #a profile

    #start by selecting a regionof sufficient altitude that the r quared law
    #will make the ratio of signal to background sufficiently low

    import numpy as np
    from copy import deepcopy

    P_out = deepcopy(P_in)
    
    if back_avg:
        P_out.values=P_out.values-back_avg
    else:    
        #select data from altitudes higher than z_min and muliply full signal by
        #range squared, if this gives less than 500 values, take uppermost 500
    
        z = P_out.index[z_min:]
    
        if len(z) <= 500:
            z = P_out.z[-500:]
    
        r_sq = P_out.values[-len(z):]*z**2
    
        #since background is constant while signal is reduced as z^2, first
        #coefficient is equal to background
        
        coeffs = np.polyfit(z,r_sq,2,full=False)
        
    
        background = coeffs[0]
    
        P_out.backsub_vals = P_out.vals-background
        P_out.back = background

    return P_out

def numdiff_d1(m_in,z):
    #generate a matrix that is the numerical differentiation of input vector
    #or w.r.t first dimension of a 2-D matrix

    import numpy as np
    from scipy import sparse

    dims = np.shape(m_in)

    data1 = np.ones(dims[0])
    data2 = -1*data1
    

    diags = [1,0]

    diffmat = sparse.spdiags([data1,data2],diags,dims[0],dims[0])

    d_m = diffmat.dot(m_in)
    
    d_z = diffmat.dot(z)
    
    dm_dz = d_m/d_z
    
    
    #set up buffer assuming dm_dz is constant for laswt two values
    
    try:
        dm_dz[-1,:] = dm_dz[-2,:]
    except IndexError:
        dm_dz[-1] = dm_dz[-2]

    return dm_dz

def numdiff_d2(m_in,x):
    #generate a matrix that is the numerical differentiation of input vector
    #or w.r.t first dimension of a 2-D matrix

    import numpy as np
    from scipy import sparse

    dims = np.shape(m_in)

    data1 = np.ones(dims[1])
    data2 = -1*data1
    

    diags = [-1,0]

    diffmat = sparse.spdiags([data1,data2],diags,dims[1],dims[1])
    
    #print diffmat

    d_m = m_in.dot(diffmat.todense())
    
    #print d_m
    
    d_x = x.dot(diffmat.todense())
    
    print d_x
    
    dm_dx = d_m/d_x
    
    #set up buffer assuming dm_dz is constant 
    
    dm_dx[:,-1] = dm_dx[:,-2]

    return dm_dx





def calc_slope(prof, winsize = 10):
    import pandas as pan
    import numpy as np
    """
    Calculates slope of data for a single profile using a smoothing window of
    predetermined size
    
    inputs:
    prof:  a pandas series where index is altitude
    n:  number of consecutive values to average
    
    output:
    slopeout: output series,same size as input,with profile slopes
    """
    data = prof.values
    altrange = np.asarray(prof.index.values,dtype='float')
    
    #Step 1: pad dataset to allow averaging
    
    leftpad = np.int(np.floor(winsize/2))
    rightpad = winsize-leftpad
      
    #Step 2: Calculate a linear fit to the data in the window
    
    slopes = np.empty(len(data)-winsize)
    for n in range(len(slopes)):       
        x = altrange[n:n+winsize]
        y = data[n:n+winsize]
        
        coeffs = np.polyfit(x,y,1,full=False)
        slopes[n] = coeffs[0]
        
    
    slopes = np.pad(slopes,(leftpad,rightpad),'edge')
    
    slope_out = pan.Series(slopes, index=altrange)
    
    
    return slope_out

def calc_SNR(prof,bg=[],bg_alt=[]):
    import pandas as pan
    import numpy as np
    """
    inputs:
    prof = a pandas series
    bg = background signal level (stray light + dark current)
    bg_alt = altitude above which signal is assumed to be purely background
             if empty, topmost 100 data points are used1
    
    Calculates signal to noise ratios for mpl data
    """
        
    if not bg_alt:
        bg_alt=prof.index[-100]
    if not bg:
        bg = np.mean(prof.ix[bg_alt])
    
    SNRprof=pan.Series(np.empty_like(prof.values),index=prof.index)
    tempvals=[v for v,r in zip(prof.values,prof.index) if r>=bg_alt]
    tempfilt=[x for x in tempvals if not np.isnan(x)]
    sigmatemp=np.std(tempfilt)
    Ctemp=sigmatemp/np.mean(np.sqrt(np.abs(tempfilt)))
    SNR = lambda x: (x-bg)/(Ctemp*np.sqrt(np.abs(x)))
        
    SNRprof[:]=np.array([SNR(v) for v in prof.values]).clip(0)
        
    return SNRprof

def boundary_layer_detect(dfin, algo="slope",slope_thresh=[],val_thresh=[],numvals=1,maxalt=2000):
    """
    Approximates the edge of the boundary layer using some combination
    of three algorithms:
    1) Negative slope threshold:  this defines the boundary layer as the lowest
    altitude the exceeds some negative slope threshold
    2) Value threshold:  this defines the top of the boundary layer as the lowest altitude
    for which the NRB dips below a value threshold
    3) Combination threshold:  Uses a combinaiton of the above two methods
    
    Inputs:
    dfin - A pandas dataframe containing a series of lidar profileswith altitude as the index
            and datetime as the column names
    algo - a string determining which algorithms to use.  Can be either:
            "slope" - purely slope threshold
            "value" - purely NRB threshold
            "combo" - combined algorithm
    slope_thresh - a floating point number defining the minimum slope to be used in the slope algorithm
    val_thresh - a floating point number defining the value threshold for the NRB value algorithm
    numvals - the number of consecutive values that must be below the slope or value threshold in order to be counted
    maxalt - an altitude in meters above which the algorithm is not longer applied
    
    Outputs:
    BL_out - a pandas series with datetime index and a boundary layer altitude in meters
    """
    import pandas as pan
    from itertools import groupby
    from operator import itemgetter
    
    
    BL_out = pan.Series(index = dfin.columns)
    
    if algo=="slope":
        for c in dfin.columns:
            tempslope = calc_slope(dfin[c])
            tempslope = tempslope.ix[:maxalt]
            for k,g in groupby(enumerate(tempslope), lambda(i,s):s>=slope_thresh):
                temp = map(itemgetter(1),g)
                if len(temp) >= numvals:
                    BL_out[c] = temp[0]
                    break
    
    if algo=="value":
        for c in dfin.columns:
            tempvals = dfin[c].ix[:maxalt]
            for k,g in groupby(enumerate(tempvals), lambda(i,v):v<=val_thresh):
                temp = map(itemgetter(1),g)
                if len(temp) >= numvals:
                    BL_out[c] = temp[0]
                    break
    
    if algo=="combo":
        for c in dfin.columns:
            tempslope = calc_slope(dfin[c])
            tempslope = tempslope.ix[:maxalt]
            tempvals = dfin[c].ix[:maxalt]
            for k,g in groupby(enumerate(tempslope), lambda(i,d):d[0]>=slope_thresh or d[1]<=val_thresh):
                temp = map(itemgetter(1),g)
                if len(temp) >= numvals:
                    BL_out[c] = temp[0]
                    break 
    
    BL_out.fillna(maxalt)
    
    return BL_out
    
def maxmin(arrayin,widths,f):
    #find all local maxima or minina from each row of an array and return array
    #of index values for each max or min
    from scipy import signal
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
        SNRprof=calc_SNR(prof)
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


    
    
    
if __name__== "__main__":

    import raman_tools as rtools
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    bkg = 0 #mV
    const = 1e-20  #combines pulse energy, Raman scattering x-sec, and optical
                    #path efficiency
    alpha_p = 5e-5 #extinction coefficient for added layer
    

    z = np.arange(100,15000,3,dtype=np.float)

    wave_0 = 532.0
    wave_r = 607.0
    

    R_mol = rtools.raman_molprof(z,wave_0,wave_r)

    z_layer = np.arange(2000,6000,5,dtype=np.float)
                     
    alpha_layer = np.ones_like(z_layer)*alpha_p

    layer = {'z':z_layer,'alpha':alpha_layer}

    R_1 = rtools.addlayer(R_mol,layer,1.0)

    P_mol = deepcopy(R_mol)
    P_mol.vals = P_mol.vals*const

    P_1 = deepcopy(R_1)
    P_1.vals = P_1.vals*const

    P_noisy = addnoise(P_1,background = bkg, stdev = 1e-4)

    P_noisy = background_subtract(P_noisy)

    rangecor_0 = np.empty_like(P_mol.vals)
    rangecor_1 = np.empty_like(P_1.vals)
    rangecor_n = np.empty_like(P_noisy.vals)

    SNR = SNR_calc(P_noisy)

    for n in range(len(z)):
        rangecor_0[n] = P_mol.vals[n]*(z[n]**2)
        rangecor_1[n] = P_1.vals[n]*(z[n]**2)
        rangecor_n[n] = P_noisy.vals[n]*(z[n]**2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(SNR,z)
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('Height [m]')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.plot(P_1.vals,z)
    ax2.set_xlabel('Lidar Return [counts]')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.plot(R_mol.T,z)
    ax3.set_xlabel('Temperature [K]')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,3,1)
    ax1.plot(rangecor_0,z,rangecor_1,z,rangecor_n,z)
    ax1.set_xscale('log')
    ax1.set_xlabel('Range corrected Raman signal multiplier')
    ax1.set_ylabel('Height [m]')

    ax2 = fig2.add_subplot(1,3,2)
    ax2.plot((R_1.alpha_t0),z)
    ax2.set_xlabel('Extinction coefficient - laser wavelength [1/m]')

    ax3 = fig2.add_subplot(1,3,3)
    ax3.plot((R_1.alpha_tr),z)
    ax3.set_xlabel('Extinction coefficient - Raman wavelength [1/m]')

    plt.show()
    
    
