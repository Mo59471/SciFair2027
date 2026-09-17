import numpy as np
from scipy.integrate import trapezoid

lAlpha, longN = np.loadtxt("LongAlpha.csv", delimiter=",", unpack=True, dtype=float)
sAlpha, shortN = np.loadtxt("ShortAlpha.csv", delimiter=",", unpack=True, dtype=float)

for i in range(len(longN)):
    longN[i] = round(longN[i])

for i in range(len(shortN)):
    shortN[i] = round(shortN[i])

#Spacing is consistent with 2/29
alphaBin = np.arange(-1.5+1/29,0.5,2/29)
alphaBin = np.append(alphaBin, alphaBin[len(alphaBin)-1]+1/29)
print(len(alphaBin),alphaBin)

sampleAlphaLong = []
sampleAlphaShort = []

#Generate a distribution of alphas by randomly sampling from the histogram; the number of random samples per alpha bin is determined by the corresponding N value
for i in range(len(alphaBin)):
    if i > 0 and i < len(alphaBin)-1:
        for j in range(int(longN[i])):
            sampleAlphaLong.append(np.random.uniform(alphaBin[i] - 2/29, alphaBin[i]))
        for j in range(int(shortN[i])):
            sampleAlphaShort.append(np.random.uniform(alphaBin[i] - 2/29, alphaBin[i]))
    else:
        for j in range(int(longN[i])):
            sampleAlphaLong.append(np.random.uniform(alphaBin[i] - 1/29, alphaBin[i]))
        for j in range(int(shortN[i])):
            sampleAlphaShort.append(np.random.uniform(alphaBin[i] - 1/29, alphaBin[i]))

# In Davis' histogram, the axes are truncated; thus the longAlpha data is missing 73 values (based on reported number of quasars vs inferred) and the short alpha data is missing 154
# to produce the offset between the obtained means and Davis' reported means:
# - Truncated long alpha objects should have avg. slope of −1.877
# - Truncated short alpha objects should have avg. slope of −1.798

for i in range(154):
    sampleAlphaShort.append(np.random.uniform(-1.798-0.3, -1.798+0.3)) # 0.6 wide range
for i in range(73):
    sampleAlphaLong.append(np.random.uniform(-1.877-0.3, -1.877+0.3)) # 0.6 wide range

print(np.mean(sampleAlphaLong),np.mean(sampleAlphaShort))
print(np.median(sampleAlphaLong),np.median(sampleAlphaShort))


#Reproduce Yoshii's integration

"""
Clean data: 
- Clean data to fit Yoshii's ranges
- Convert wavelengths to frequencies
"""

#Every 241 freqlengths assigned to 1 radius
freq, Qabs = np.genfromtxt("Gra_81.txt", skip_header=5, usecols=(0,1), delimiter=",", unpack=True, dtype=None)

delIndex = []
radii = []
for i in range(len(Qabs)):
    if Qabs[i] == "=":
        radii.append(float(freq[i]))
    if Qabs[i] == "=" or Qabs[i] == "Q_abs":
        delIndex.append(i)

freq = np.delete(freq, delIndex)
Qabs = np.delete(Qabs, delIndex)

freq = freq.astype(float)
Qabs = Qabs.astype(float)

for i in range(len(freq)):
    #Convert wavelengths in microns -> angstroms
    freq[i] = 10000 * freq[i]

#Get rid of wavelengths outside of valid range: UV/optical
#For the IR integral, emission across all wavelengths is relative: No trimming necessary
delIndex_long = []
delIndex_short = []
for i in range(len(freq)):
    if freq[i] > 5500 or freq[i] < 2200:
        delIndex_long.append(i)
    if freq[i] > 2200 or freq[i] < 300:
        delIndex_short.append(i)
        
freqUV_long = np.delete(freq, delIndex_long)
QabsUV_long = np.delete(Qabs, delIndex_long)
freqUV_short = np.delete(freq, delIndex_short)
QabsUV_short = np.delete(Qabs, delIndex_short)

#Convert wavelengths to frequencies
for i in range(len(freq)):
    freq[i] = 299792458/(freq[i] * 1e-10)
for i in range(len(freqUV_long)):
    freqUV_long[i] = 299792458/(freqUV_long[i] * 1e-10)
for i in range(len(freqUV_short)):
    freqUV_short[i] = 299792458/(freqUV_short[i] * 1e-10)
    
#Bin each frequency in a list corresponding to the radius
ogFreqUV_long = freqUV_long
ogQabsUV_long = QabsUV_long
ogFreqUV_short = freqUV_short
ogQabsUV_short = QabsUV_short
ogFreqIR = freq
ogQabsIR = Qabs

freqUV_long = []
QabsUV_long = []
freqUV_short = []
QabsUV_short = []
freqIR = []
QabsIR = []

for i in range(len(ogFreqUV_long)):
    if ogFreqUV_long[i] == np.min(ogFreqUV_long):
        freqUV_long.append([ogFreqUV_long[i]])
        QabsUV_long.append([ogQabsUV_long[i]])
    else:
        freqUV_long[len(freqUV_long)-1].append(ogFreqUV_long[i])
        QabsUV_long[len(QabsUV_long)-1].append(ogQabsUV_long[i])

for i in range(len(ogFreqUV_short)):
    if ogFreqUV_short[i] == np.min(ogFreqUV_short):
        freqUV_short.append([ogFreqUV_short[i]])
        QabsUV_short.append([ogQabsUV_short[i]])
    else:
        freqUV_short[len(freqUV_short)-1].append(ogFreqUV_short[i])
        QabsUV_short[len(QabsUV_short)-1].append(ogQabsUV_short[i])

for i in range(len(ogFreqIR)):
    if ogFreqIR[i] == np.min(ogFreqIR):
        freqIR.append([ogFreqIR[i]])
        QabsIR.append([ogQabsIR[i]])
    else:
        freqIR[len(freqIR)-1].append(ogFreqIR[i])
        QabsIR[len(QabsIR)-1].append(ogQabsIR[i])

#Clean radii outside of Yoshii's range: aMin = 0.005 microns, aMax = 0.2 microns

QabsUV_long = QabsUV_long[14:47]
freqUV_long = freqUV_long[14:47]

QabsUV_short = QabsUV_short[14:47]
freqUV_short = freqUV_short[14:47]

QabsIR = QabsIR[14:47]
freqIR = freqIR[14:47]

radii = radii[14:47]

#Note; all frequencies are the same for each radius, Qabs is the only thing that changes

"""INTEGRATION PROCEDURE"""

# alphaS = -0.5
# alphaL = -0.4

g = []
sampledAlphaL = []
sampledAlphaS = []

TsubTog = False #Toggle whether or not to sample sublimation temperature; this is kind of iffy, since Tsub is mostly universal(material dependent); however, certain object-to-object environmental variations can change this

#Sample 10000 iterations of g
for i in range(1000):
    
    if TsubTog:
        Tsub = np.random.normal(1700,50)
    else:
        Tsub = 1700
    
    #Numerator double integral

    #Equation for blackbody spectrum as a function of frequency
    blackBodyIR = ( #Gives you the blackbody spectrum as a function of freqIR[0]: First element in freqIR = all tabulated frequencies (no truncation)
        2*6.6260693e-34*np.asarray(freqIR[0])**3/299792458**2
        * 1/(np.e**(6.6260693e-34*np.asarray(freqIR[0])/(1.380658e-23*Tsub))-1)
    )

    #Integrate Qabs over only the frequencies (multidimensional array) -> returns integral of Qabs as a function of grain size
    numInt1_IR = trapezoid(QabsIR*blackBodyIR,freqIR[0])
    numInt_IR = trapezoid(np.asarray(radii)**-0.75 * numInt1_IR, radii)
    
    #Assuming low covariance (independent alpha short, alpha long):
    alphaS = np.random.choice(sampleAlphaShort)
    alphaL = np.random.choice(sampleAlphaLong)
    
    # #This gives: 10.438382847472575
    # alphaS= np.median(sampleAlphaShort)
    # alphaL = np.median(sampleAlphaLong)
    
    # #This gives: 10.558182968737368
    # alphaS= np.mean(sampleAlphaShort)
    # alphaL = np.mean(sampleAlphaLong)
    
    #Monte Carlo gives: 10.460683919706051 (median), 10.500281070284975 (mean)
    
        
    sampledAlphaL.append(alphaL)
    sampledAlphaS.append(alphaS)
    
    numIntLong_UV = trapezoid(QabsUV_long*(np.asarray(freqUV_long[0])/(299792458/(5500 * 1e-10)))**alphaL,freqUV_long[0])
    numIntShort_UV = trapezoid(QabsUV_short*(np.asarray(freqUV_short[0])/(299792458/(2200 * 1e-10)))**alphaS * (5500/2200)**alphaL,freqUV_short[0])
    numInt_UV = trapezoid(np.asarray(radii)**-0.75 * (numIntLong_UV + numIntShort_UV), radii)

    g.append(2.5 * np.log10(4 * np.pi * (299792458 *2.80003e-13) ** 2 * numInt_IR/(numInt_UV*3640*1e-26)))
            
"""Gives: 
- At fixed alpha = -0.5: 10.421961246662185S
    - Offset from yoshii by 10.6 - 10.421961246662185 = 0.1780387533
- 10.251119793399347 with different freq_long vs freq_short (alphaS = -0.5, alphaL = -0.4)

which slightly varies from Yoshii, likely due to differing methodology specifics (e.g. different radius-weighting)

#IDEA: Additive mapping from this value to Yoshii's (translates to multiplicative mapping since g is in log scales)

"""

for i in range(len(g)):
    g[i] = g[i] + 0.1780387533

with open("SampledG.txt", 'w') as file:
    for i in range(len(g)):
        file.write(f"{g[i]}|{sampledAlphaL[i]}|{sampledAlphaS[i]}\n")

print(np.median(g))
print(np.mean(g))
print(np.std(g))