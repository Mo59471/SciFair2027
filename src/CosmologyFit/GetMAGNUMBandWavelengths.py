import numpy as np
import math
from scipy.integrate import quad

zHel, zCMB, tau_mag, tau_mag_err, tau_mag_Err, logLmag, logLmag_err, flux_MAG, flux_MAG_Err, flux_MAG_err, obsBand = np.genfromtxt(
    "MAGNUMData.csv", delimiter=",", usecols = (1,2,5,6,7,10,13,14,15,16,20), 
    skip_header=1, unpack = True, missing_values=(""), filling_values=-999, dtype=None
)

R_frq = []
I_frq = []
R_wave = []
I_wave = []

def integrand(z):
    return 1/np.sqrt(0.27 * (1+z)**3 + 0.73)

for i in range(len(zHel)):
    if flux_MAG[i] != -999:
        DL = (1+zHel[i]) * 3.0856776e22 * 299792.458/73 * quad(integrand, 0, zHel[i])[0]
        frqX = (299792458/(5500*1e-10)) * (
            (flux_MAG[i]*1e-29*4*math.pi*DL**2)/
            (10**logLmag[i]*1e-7*5500*1e-10/299792458*(1+zHel[i])**0.7)
            )**(1/-0.3)
        waveX = 299792458/frqX * 1e10
        
        if str(obsBand[i]) == "R":
            R_frq.append(frqX)
            R_wave.append(waveX)
            print(waveX, obsBand[i])
        elif str(obsBand[i]) == "I":
            I_frq.append(frqX)
            I_wave.append(waveX)
            print(waveX, obsBand[i])
    
print(np.mean(R_frq), np.mean(R_wave))
print(np.mean(I_frq), np.mean(I_wave))
