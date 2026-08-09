#MC sampler that samples corrected time lags using a power law index determined from comparison of W1 and W2 time lags
#Samples from the time lag error margins to produce gammas and corrected time lags
#Produces 10,000, sampled, corrected time lags for each object

from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from astropy import units as u
import numpy as np
import random
import scipy
import statistics
import math
import matplotlib.pyplot as plt
import mpfit

#Paramaters

objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
    "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
    skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None) #-999 indicates no value

zCMB = []
zFlowCorr = []

#Cosmology Parameters
H0 = []

#Luminosity distances
DL_given = []
DL_tau = []
DL_err = []

flux = []
flux_err = []
flux_Err = []

#Lists of raw time lags (Mandal's correction with gamma = 0.62 removed)
rTau = []
rTau_err = []
rTau_Err = []

#Lists of sampled time lags
sTau = []

mode= [] #Convert from the median time lag value (reported by Mandal) to modes for split-normal sampling

#Extract RA and Dec from SDSS IDs
RA = []
Dec = []

for obj in SDSSID:
    RAH = float(obj[:2])
    RAM = float(obj[2:4])
    if "+" in obj:
        RAS = float(obj[4:obj.index("+")])
        DecD = float(obj[obj.index("+")+1:obj.index("+")+3])
        DecM = float(obj[obj.index("+")+3:obj.index("+")+5])
        DecS = float(obj[obj.index("+")+5:])
    else:
        RAS = float(obj[4:obj.index("-")])
        DecD = float(obj[obj.index("-")+1:obj.index("-")+3])
        DecM = float(obj[obj.index("-")+3:obj.index("-")+5])
        DecS = float(obj[obj.index("-")+5:])
    
    RA.append(15*(RAH + (RAM/60) + (RAS/3600)))
    if "-" in obj:
        Dec.append(-1*(DecD + (DecM/60) + (DecS/3600)))
    else:
        Dec.append(DecD + (DecM/60) + (DecS/3600))
        
#Calculate CMB rest frame redshift based on coordinates and zHel
lat = [] #Galactic latitude and longitude 
long = []

for coord1, coord2 in zip(RA,Dec):
    coord = SkyCoord(ra=float(coord1)*u.degree, dec = float(coord2)*u.degree, frame = "icrs") #Convert to galactic coordinates
    galCoord = coord.galactic
    lat.append(galCoord.b.deg)
    long.append(galCoord.l.deg)
for i in range(len(zHel)):
    #Uses the formula given by Peterson et al. 2022: Dipole direction and sun velocity relative to CMB rest frame is from the Planck Collab
    zCMB.append(
        (1+float(zHel[i]))/
        (1-((369.82/299792.458)*
            (np.sin(np.radians(lat[i]))*
                np.sin(np.radians(48.253))+
                np.cos(np.radians(lat[i]))*
                np.cos(np.radians(48.253))*
                np.cos(np.radians(long[i]-264.021)))))
        -1)

#Get raw time lags
for i in range(len(zCMB)):
    if zCMB[i] >= 0.5454545455 and zCMB[i] <= 0.6:
        if tauW1[i] != -999:
            rTau.append(tauW1[i]/(1+zReported[i])**-0.38)
            rTau_err.append(tauW1_err[i]/(1+zReported[i])**-0.38)
            rTau_Err.append(tauW1_Err[i]/(1+zReported[i])**-0.38)
        else:
            rTau.append(-999)
            rTau_Err.append(-999)
            rTau_err.append(-999)
    elif zCMB[i] >= 1.090909091 and zCMB[i] <= 1.145454546:
        if tauW2[i] != -999:
            rTau.append(tauW2[i]/(1+zReported[i])**-0.38)
            rTau_err.append(tauW2_err[i]/(1+zReported[i])**-0.38)
            rTau_Err.append(tauW2_Err[i]/(1+zReported[i])**-0.38)
        else:
            rTau.append(-999)
            rTau_Err.append(-999)
            rTau_err.append(-999)
    else:
        rTau.append(-999)
        rTau_Err.append(-999)
        rTau_err.append(-999)

#Obtain mode time lags from medians assuming a split-normal distribution
for i in range(len(rTau)):
    if rTau[i] != -999:
        if rTau_err[i] > rTau_Err[i]: #Left skewed
            mode.append(rTau[i] - rTau_err[i] * scipy.stats.norm.ppf((rTau_err[i] + rTau_Err[i])/(4*rTau_err[i]))) #Calculate mode from median based on skew
        elif rTau_err[i] < rTau_Err[i]: #Right skewed
            mode.append(rTau[i] - rTau_Err[i] * scipy.stats.norm.ppf(1-((rTau_err[i] + rTau_Err[i])/(4*rTau_Err[i]))))
        elif rTau_err[i] == rTau_Err[i]: #Symmetric
            mode.append(rTau[i])
    else:
        mode.append(-999)          

#Sample randomly from the split normal distributions of the time lags
#Here, the uncertainties between W1 and W2 are assumed to be independent which is not strictly true, but approximately so
def sampleSplitNormal(mode, sigmaL, sigmaR):
    pLeft = sigmaL/(sigmaL+sigmaR) #Define cumulative probability of LH side
    if random.uniform(0,1) <= pLeft: #random float between 0-1, inclusive; if less than prob of LH side -> falls in LH side
        return -1*abs(random.gauss(0, sigmaL))+mode #If on left hand side, sample from LH Gaussian
    else:
        return abs(random.gauss(0,sigmaR))+mode #If on right hand side, sample from RH Gaussian

#Run 10,000 sampling realizations to get 10,000 corrected time lags for each object
for i in range(10000):
    for j in range(len(mode)):
        if mode[j] != -999:
            lag = sampleSplitNormal(mode[j], rTau_err[j], rTau_Err[j])
            while lag <= 0: #If less than 0, repeat sampling
                lag = sampleSplitNormal(mode[j], rTau_err[j], rTau_Err[j])
                            
            #Append the raw sampled value to lists of lists (each sublist = distribution of sampled values for each object)
            if len(sTau) != len(mode):
                sTau.append([(lag/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18)])
            else:
                sTau[j].append((lag/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18))             
        else:
            if len(sTau) != len(mode):
                sTau.append([-999])
    
#Get luminosity distances using the adopted cosmology of Shen and Liu
def integrand(z):
    return 1/np.sqrt((0.3*(1+z)**3+0.7))

for i in range(len(zCMB)):
    if sTau[i][0] != -999:
        integral = quad(integrand, 0, zCMB[i])
        DL_given.append(3.0856776e24*(1+zCMB[i])*299792.458/70*integral[0]) #Large conversion factor in front to convert Mpc to cm
    else:
        DL_given.append(-999)

#Get fluxes from luminosity distances 
for i in range(len(logL)):
    if sTau[i][0] != -999:
        if zCMB[i] < 0.7: #Correct back to spectral luminosity in erg/s/Hz
            L = 10**logL[i]/(299792458/(5100*1e-10))
        elif zCMB[i] < 1.9:
            L = 10**logL[i]/(299792458/(3000*1e-10))
        else:
            L = 10**logL[i]/(299792458/(1350*1e-10))
        L_err = 10**math.log10(L)-10**(math.log10(L)-logL_err[i])
        L_Err = 10**(math.log10(L)+logL_err[i])-10**math.log10(L)
        flux.append(L/(4*math.pi*DL_given[i]**2)) #erg/s/Hz/cm^2
        flux_err.append(L_err/(4*math.pi*DL_given[i]**2))
        flux_Err.append(L_Err/(4*math.pi*DL_given[i]**2))
    else:
        flux.append(-999)
        flux_err.append(-999)
        flux_Err.append(-999)
    
#Get distances from Minezaki's formula given time lags
#Temporarily, alphaUV is adopted as -0.5 (following Yoshii)
#Future implementation of an MC sampling from alphaUV distributions to come
def sampleLogNormal(mu, sigma): #Sample flux values from lognormal distribution
    mu = mu/math.log10(math.e) #Convert log10 mode and sigma to ln for lognormvariate (base change theorem)
    sigma = sigma/math.log10(math.e)
    randF = random.lognormvariate(mu, sigma)
    while randF <= 0:
        randF = random.lognormvariate(mu, sigma)
    return randF

#10,000 realizations of MC sampled time lags and fluxes to obtain DL
for i in range(len(sTau)):
    if sTau[i][0] != -999:
        if zCMB[i] < 0.7:
            kCorr = -2.5 * -0.5 * math.log10(5500/5100)
        elif zCMB[i] < 1.9:
            kCorr = -2.5 * -0.5 * math.log10(5500/3000)
        else:
            kCorr = -2.5 * -0.5 * math.log10(5500/1350)
        DL_tau.append([])
        for j in range(10000):
            mV = -2.5 * math.log10(sampleLogNormal(math.log10(flux[i]),logL_err[i])/(3640*1e-23)) #Assume a normalization of 3640 Jy, converted to erg/s/cm^2/Hz by factor of 10^-23
            DL_tau[i].append(sTau[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
    else:
        DL_tau.append([-999])
        
#Fit cosmology: bootstrapping + MC resampling
#Least squares residual minimization using MPFIT

#Cleaned MC sampled luminosity distance lists (-999 removed)
DL_tau_clean = []
zCMB_clean = []

for i in range(len(DL_tau)):
    if DL_tau[i][0] != -999:
        DL_tau_clean.append(DL_tau[i])
        DL_err.append(np.std(DL_tau[i]))
        zCMB_clean.append(zCMB[i])

#Fit cosmology with MPFIT
#Minimize residual DLs by calculating the luminosity distance integral

#Construct the cosmological model to be iterated upon (DL integral)
def cosmoModel(z, H0, omg_0, zStepNum):
    z = np.asarray(z, dtype= float)
    zSteps = np.linspace(0,np.max(z), zStepNum) #Gives an array of equally spaced z values up to the max for numeric integration
    
    denom = omg_0*(1+zSteps)**3 + (1-omg_0) #omg_DE in flat cosmology
    
    #Calculatges a trapezoidal-cumulative integral up to the max redshift using the integrand (y(x)), and evenly spaced x step values (zSteps)
    runningIntegral = cumulative_trapezoid(1/np.sqrt(denom),zSteps,initial = 0.0)
    
    #There now is a cumulative integral up to the max redshift, with values for each zStep
    #Next, for each value in the given z, linearly interpolate between the z-step integral vlaues and approximate the integral's value at the given z
    integrals = np.interp(z,zSteps,runningIntegral) #gives np array of integrals at different z values
    
    return (1+z) * 299792.458/H0 *integrals

def cosmoFit(p, fjac = None, z = None, DL = None, err = None):
    z = np.asarray(z,dtype=float)
    DL = np.asarray(DL, dtype=float)
    
    model = cosmoModel(z,p[0],0.3,1000) 
    #status = 0
    return [0, (DL-model)/err] 

p0 = [70] #Starting parameter estimates: reduces computational time to list them close to true values, but does not bias result

"""
p[0]: Hubble Constant
Dimensionless density parameter for matter fixed at 0.3
"""
for i in range(10000):
    DL_fit = []
    zCMB_fit = []
    while len(DL_fit) < len(DL_tau_clean):
        #Bootsraps: randomly selects objects (with replacement) until the boostrapped sample is as large as the original
        index = random.randint(0,len(DL_tau_clean)-1)
        #MC sampling: randomly selects a value from the random object's DL uncertainty distribution
        DL_fit.append(random.choice(DL_tau_clean[index]))
        zCMB_fit.append(zCMB_clean[index])
    
    #Set limits for valid cosmology parameters
    parinfo = [
        {
            "limited": [1, 0], #Limited only on bottom side, open on top side
            "limits": [1.0,0.0] #Limit H0 to a min of 1.0: prevent negative or 0 H0
        }
    ]
    
    m= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_fit, dtype=float), "DL":np.array(DL_fit), "err": np.array(DL_err)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1 #Turn off console logs
        )


    H0.append(float(m.params[0]))

print(statistics.median(H0))

DL_tau_meds = []

for i in range(len(DL_tau_clean)):
    DL_tau_meds.append(statistics.median(DL_tau_clean[i]))

plt.ion()

plt.xscale("log")
plt.yscale("log")
plt.xlim(0.53,1.2)
plt.ylim(1e3,25000)
        
plt.plot(zCMB_clean,DL_tau_meds, "bo", ms = 1)

def getDLPlot(H0 = None):
    if H0 is None:
        H0 = 73.04
    else:
        H0 = statistics.median(H0)
    return cosmoModel(np.linspace(0,3,100),H0,0.27,1000)

plt.plot(np.linspace(0,3,100),getDLPlot(),ls = "--",  color = "orange")
plt.plot(np.linspace(0,3,100),getDLPlot(H0),  color = "green")

plt.show()

plt.figure()
histH0, binH0, _ = plt.hist(H0, bins = "auto")
plt.stairs(histH0,binH0,edgecolor = "blue")

plt.show(block = True)

print("close")

            