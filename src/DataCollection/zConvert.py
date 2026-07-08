from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

RA = []
Dec = []
lat = []
long = []
lb = []
zCMB = []
zWrite = []
toggle = 1

if toggle == 0:
    with open("WISECoordsData.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            RA.append(line[:line.index("|")])
            Dec.append(line[line.index("|")+1:])
        for coord1,coord2 in zip(RA,Dec):
            coord = SkyCoord(ra=float(coord1)*u.degree, dec = float(coord2)*u.degree, frame = "icrs")
            galCoord = coord.galactic
            lat.append(galCoord.b.deg)
            long.append(galCoord.l.deg)
            lb.append(f"{galCoord.l.deg}|{galCoord.b.deg}\n")
    with open("WISERedshifts.txt", "r") as file:
        zHel = file.readlines()
        for i in range(len(zHel)):
            #Uses the formula given by Peterson et al. 2022: Dipole direction and sun velocity relative to CMB rest frame is from the Planck Collab
            zCMB.append((1+float(zHel[i]))/(1-((369.82/299792.458)*(np.sin(np.radians(lat[i]))*np.sin(np.radians(48.253))+np.cos(np.radians(lat[i]))*np.cos(np.radians(48.253))*np.cos(np.radians(long[i]-264.021)))))-1)
    for i in zCMB:
        zWrite.append(f"{i}\n")      
    with open("WISERedshiftsCMB.txt", "w") as file:
        file.writelines(zWrite)

else:
    with open("MAGNUMCoordsDeg.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            RA.append(line[:line.index("|")])
            Dec.append(line[line.index("|")+1:])
        for coord1,coord2 in zip(RA,Dec):
            coord = SkyCoord(ra=float(coord1)*u.degree, dec = float(coord2)*u.degree, frame = "icrs")
            galCoord = coord.galactic
            lat.append(galCoord.b.deg)
            long.append(galCoord.l.deg)
            lb.append(f"{galCoord.l.deg}|{galCoord.b.deg}\n")
    with open("MAGNUMRedshifts.txt", "r") as file:
        zHel = file.readlines()
        for i in range(len(zHel)):
            #Uses the formula given by Peterson et al. 2022: Dipole direction and sun velocity relative to CMB rest frame is from the Planck Collab
            zCMB.append((1+float(zHel[i]))/(1-((369.82/299792.458)*(np.sin(np.radians(lat[i]))*np.sin(np.radians(48.253))+np.cos(np.radians(lat[i]))*np.cos(np.radians(48.253))*np.cos(np.radians(long[i]-264.021)))))-1)
    for i in zCMB:
        zWrite.append(f"{i}\n")  
    with open("MAGNUMRedshiftsCMB.txt", "w") as file:
        file.writelines(zWrite)

with open("test.txt", "w") as file:
    file.writelines(lb)      
        
