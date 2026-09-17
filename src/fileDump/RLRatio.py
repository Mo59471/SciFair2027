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

objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
    "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
    skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None) #-999 indicates no value

tauW1_clean = []
tauW2_clean = []

zCMB = []
zW1 = []
zW2 = []
DL_given = []
DL_tauW1 = []
DL_tauW2 = []
flux = []


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

for i in range(len(tauW1)):
    if tauW1[i] != -999:
        tauW1_clean.append(tauW1[i])
        zW1.append(zCMB[i])
    if tauW2[i] != -999:
        tauW2_clean.append(tauW2[i])
        zW2.append(zCMB[i])
    
#Get luminosity distances using the adopted cosmology of Shen and Liu
def integrand(z):
    return 1/np.sqrt((0.3*(1+z)**3+0.7))

for i in range(len(zCMB)):
    integral = quad(integrand, 0, zCMB[i])
    DL_given.append(3.0856776e24*(1+zCMB[i])*299792.458/70*integral[0]) #Large conversion factor in front to convert Mpc to cm

#Get fluxes from luminosity distances 
for i in range(len(logL)):
    if zCMB[i] < 0.7: #Correct back to spectral luminosity in erg/s/Hz
        L = 10**logL[i]/(299792458/(5100*1e-10))
    elif zCMB[i] < 1.9:
        L = 10**logL[i]/(299792458/(3000*1e-10))
    else:
        L = 10**logL[i]/(299792458/(1350*1e-10))
    L_err = 10**math.log10(L)-10**(math.log10(L)-logL_err[i])
    L_Err = 10**(math.log10(L)+logL_err[i])-10**math.log10(L)
    flux.append(L/(4*math.pi*DL_given[i]**2)) #erg/s/Hz/cm^2

#Get DL from time lags
for i in range(len(tauW1_clean)):
    if tauW1_clean[i] != -999:
        if zW1[i] < 0.7:
            kCorr = -2.5 * -0.5 * math.log10(5100/5500)
        elif zW1[i] < 1.9:
            kCorr = -2.5 * -0.5 * math.log10(3000/5500)
        else:
            kCorr = -2.5 * -0.5 * math.log10(1350/5500)
        mV = -2.5 * math.log10(flux[i]/(3640*1e-23)) #Assume a normalization of 3640 Jy, converted to erg/s/cm^2/Hz by factor of 10^-23
        DL_tauW1.append(tauW1_clean[i] * 10**(0.2*(mV-kCorr-25+10.60)))

for i in range(len(tauW2_clean)):
    if tauW2_clean[i] != -999:
        if zW2[i] < 0.7:
            kCorr = -2.5 * -0.5 * math.log10(5100/5500)
        elif zW2[i] < 1.9:
            kCorr = -2.5 * -0.5 * math.log10(3000/5500)
        else:
            kCorr = -2.5 * -0.5 * math.log10(1350/5500)
        mV = -2.5 * math.log10(flux[i]/(3640*1e-23)) #Assume a normalization of 3640 Jy, converted to erg/s/cm^2/Hz by factor of 10^-23
        DL_tauW2.append(tauW2_clean[i] * 10**(0.2*(mV-kCorr-25+10.60)))

def getDLPlot(z, H0, omg_0):
    return(
        (1+z) * 
        299792.458/H0 * 
        cumulative_trapezoid(
            1/np.sqrt(omg_0* 
            (1+z)**3 + (1-omg_0)),
            z,initial = 0.0)
            )

fig, ax = plt.subplots(1,2)

zPlot = np.linspace(0,np.max(zCMB),1000)

for i in range(len(ax)):
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")

ax[0].plot(zW1, DL_tauW1, "bo", ms = 1)
ax[1].plot(zW2, DL_tauW2, "ro", ms = 1)

for i in ax:
    i.plot(zPlot, getDLPlot(zPlot, 73.04, 0.27), color = "green")

plt.show()