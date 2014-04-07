def backandnoise(P_in,background = 0,stdev = []):
    #Adds gaussian random noise and background signal to any profile
    #if no standard deviation is defined, the noise added is standard
    #shot noise - poisson distrobution approximated by a gaussian with
    #std = sqrt(signal)
    
    import numpy as np
    import random
    
    P_out = P_in+background
    
    if stdev:
        for n in range(len(P_in)):
            P_out[n] = P_out[n] + random.gauss(background,stdev)
            #print random.gauss(background,stdev)
    else:
        for n in range(len(P_in)):
            stdev = np.sqrt(P_in[n]+background)
            P_out[n] = P_in[n] + random.gauss(background,stdev)

    return P_out

def background_subtract(P_in):
    #takes advantage of inverse square law to calculate background signal for
    #a profile

    #start by selecting a regionof sufficient altitude that the r quared law
    #will make the ratio of signal to background sufficiently low

    import numpy as np
    from copy import deepcopy

    P_out = deepcopy(P_in)
    
    z_min = 6000 #[m]

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

    
def SNR_1D(P_in,winsize=10):
    #estimates signal to noise ratio from an input profile
    #as a function of altitude using a sliding window of 100 pixels assuming for this local section
    #a second-order polynomial curve fit is acceptable to fit the mean value
    
    import numpy as np

    z = P_in.index.values
    
    try:
        vals = P_in.values
    except AttributeError:
        print 'Warning: Background subtraction has not been performed prior to SNR calculation'
        vals = P_in.values
    
    stdev = np.empty_like(vals)
    
    #create buffer around 
    
    for n in range(len(vals)-winsize):
        z_win = z[n:(n+winsize)]
        v_win = vals[n:(n+winsize)]
    
        coeffs = np.polyfit(z_win,v_win,2,full=False)
        
        baseline = coeffs[0]*z_win**2 + coeffs[1]*z_win + coeffs[2]
        
        noise = v_win-baseline
        
        stdev[n] = np.std(noise)
        
    stdev[n:] = stdev[n]
    
    SNR = vals/stdev
    
    return SNR


def SNR_2D (dfin,boxsize = (10,10)):
    import pandas as pan
    import numpy as np
    """
    inputs:
    dfin = a pandas dataframe
    boxsize=(10,10)  a tuple defining the size of the box to calculate in (x,y)
    
    Takes in a pandas dataframe and generates a dataframe of the same size containing
    2-D signal to noise ratio calculations.
    """
    
    #create buffer around dataframe
    data = dfin.values
    (rows,columns) = dfin.shape
    newsize = (rows+boxsize[0],columns+boxsize[1])
    newarray = np.empty(newsize)
    (newrows,newcolums) = newarray.shape
    
    l = int(np.ceil(boxsize[0]/2))
    r = boxsize[0]-leftbuffer
    t = int(np.ceil(boxsize[1]/2))
    b = boxsize[1]-topbuffer
    
    #create buffered array for calculating mean and std
    
    newarray[:l,t:-b] = data[:l,:]
    newarray[l:-r,t:-b] = data
    newarray[-r:,t:-b] = data[-r:,:]
    newarray[:,:t] = newarray[:,t:2*t]
    newarray[:,-b:] = newarray[:,-2*b:-b]
    
    #calculate SNR from mean and std
    SNRout = np.empty_like(data)
    for r in range(rows):
        for c in range(columns):
            window = newarray[r:r+boxsize[0],c:c+boxsize[1]]
            tempmean = np.mean(window)
            tempstd = np.std(window)
            SNRout[r,c] = tempmean/tempstd    
    
    dfout = pan.DataFrame(data = SNRout, index = dfin.index, columns = dfin.coulmns)
    
    return dfout

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
    
#def layer_detect(dfin, algo="slope",slope_thresh=[],val_thresh=[],numvals=1,altrange=[]):
    """
    Takes a pandas series showing a lidar profile and uses a combination of
    slopes and threshold levels to detect feature edges
    
    inputs:
    dfin: a pandas dataframe wit datetime index and altitude columns
    slope=[]: if slope is defined, this value is used as the threshold slope to demarcate layers
    thresold=[] if 
    
    """
    
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
    
    
