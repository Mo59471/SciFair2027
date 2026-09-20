"""
Calculate and tabulate grid of Kernel Density Estimations, used for obtaining likelihoods in the Bayesian fit

- Cutdown runtime: Don't have calculate grids while fitting, simply read from files

"""
import numpy as np
from scipy.stats import gaussian_kde # Kernel density estimation for smoothing histograms

# WISE data
objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
    "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
    skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None
)

# MAGNUM data
zCMB_mag, tauK, tauK_err, tauK_Err, logLmag, logLmag_err, flux_mag, flux_mag_Err, flux_mag_err, obsBand = np.genfromtxt(
    "MAGNUMData.csv", delimiter=",", usecols = (2,5,6,7,10,13,14,15,16, 20), 
    skip_header=1, unpack = True, missing_values=(""), filling_values=-999, dtype=None
)

# WISE redshifts
zCMB_WISE = np.genfromtxt(
    "WISE_RedshiftsCMB.txt", delimiter="|",unpack = True, dtype=float 
)

# Luminosity distances: 2D arrays with shape[0] = object, shape[1] = sample

# W1 distances
DL_tauW1 = np.genfromtxt(
    "LuminosityDistances_W1.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW1_118 = np.genfromtxt(
    "LuminosityDistances_W1_118.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW1_28 = np.genfromtxt(
    "LuminosityDistances_W1_28.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW1_WISE = np.genfromtxt(
    "LuminosityDistances_W1_WISE.txt", delimiter = "|", unpack = True, dtype=float 
)

# W2 distances
DL_tauW2 = np.genfromtxt(
    "LuminosityDistances_W2.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_118 = np.genfromtxt(
    "LuminosityDistances_W2_118.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_28 = np.genfromtxt(
    "LuminosityDistances_W2_28.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_WISE = np.genfromtxt(
    "LuminosityDistances_W2_WISE.txt", delimiter = "|", unpack = True, dtype=float 
)

# K distances
DL_tauK = np.genfromtxt(
    "LuminosityDistances_K.txt", delimiter = "|", unpack = True, dtype=float 
)

"""
Data lists

"""

# Cleaned luminosity distance lists (-999 removed, in linear Mpc)
DL_tauW1_clean = []
DL_tauW2_clean =[]
DL_tauW1_118_clean = []
DL_tauW2_118_clean = []
DL_tauW1_28_clean = []
DL_tauW2_28_clean = []
DL_tauW1_WISE_clean = []
DL_tauW2_WISE_clean = []
DL_tauK_clean = []

# Clean CMB redshifts (values without corresponding distance removed)
zCMB_W1 = []
zCMB_W2 = []
zCMB_K = []

"""
Get cleaned distance lists, get cleaned redshift lists

"""

# W1 objects
for i in range(len(DL_tauW1)):
    
    if DL_tauW1[i][0] != -999:
        DL_tauW1_clean.append(DL_tauW1[i])
        DL_tauW1_118_clean.append(DL_tauW1_118[i])
        DL_tauW1_28_clean.append(DL_tauW1_28[i])
        DL_tauW1_WISE_clean.append(DL_tauW1_WISE[i])

        zCMB_W1.append(zCMB_WISE[i])

# W2 objects
for i in range(len(DL_tauW2)):
    
    if DL_tauW2[i][0] != -999:
        DL_tauW2_clean.append(DL_tauW2[i])
        DL_tauW2_118_clean.append(DL_tauW2_118[i])
        DL_tauW2_28_clean.append(DL_tauW2_28[i])
        DL_tauW2_WISE_clean.append(DL_tauW2_WISE[i])

        zCMB_W2.append(zCMB_WISE[i])

# K objects  
for i in range(len(DL_tauK)):
    if DL_tauK[i][0] != -999:
        DL_tauK_clean.append(DL_tauK[i])
        zCMB_K.append(zCMB_mag[i])
    
"""
Get KDEs
    
"""

# Get the distributions of DL in logarithmic space
logDL_tauK = np.log10(DL_tauK_clean)
logDL_tauW1 = np.log10(DL_tauW1_clean)
logDL_tauW1_118 = np.log10(DL_tauW1_118_clean)
logDL_tauW1_28 = np.log10(DL_tauW1_28_clean)
logDL_tauW1_WISE = np.log10(DL_tauW1_WISE_clean)
logDL_tauW2 = np.log10(DL_tauW2_clean)
logDL_tauW2_118 = np.log10(DL_tauW2_118_clean)
logDL_tauW2_28 = np.log10(DL_tauW2_28_clean)
logDL_tauW2_WISE = np.log10(DL_tauW2_WISE_clean)

# Convert to smoothed porbability distributions using Kernel Denisty Estimation

# Lists of KDE objects (distributions autonormalized by SciPy to become PDFs)
kdeDL_tauK = []
kdeDL_tauW1 = []
kdeDL_tauW1_118 = []
kdeDL_tauW1_28 = []
kdeDL_tauW1_WISE = []
kdeDL_tauW2 = []
kdeDL_tauW2_118 = []
kdeDL_tauW2_28 = []
kdeDL_tauW2_WISE = []

# K objects
for i in range(len(logDL_tauK)):
    kde = gaussian_kde(
        logDL_tauK[i],
        bw_method="scott"
    )
    kdeDL_tauK.append(kde)

# W1 objects
for i in range(len(logDL_tauW1)):
    kde = gaussian_kde(
        logDL_tauW1[i],
        bw_method="scott"
    )
    kde118 = gaussian_kde(
        logDL_tauW1_118[i],
        bw_method="scott"
    )
    kde28 = gaussian_kde(
        logDL_tauW1_28[i],
        bw_method="scott"
    )
    kdeWISE = gaussian_kde(
        logDL_tauW1_WISE[i],
        bw_method="scott"
    )
    
    kdeDL_tauW1.append(kde)
    kdeDL_tauW1_118.append(kde118)
    kdeDL_tauW1_28.append(kde28)
    kdeDL_tauW1_WISE.append(kdeWISE)

# W2 objects
for i in range(len(logDL_tauW2)):
    kde = gaussian_kde(
        logDL_tauW2[i],
        bw_method="scott"
    )
    kde118 = gaussian_kde(
        logDL_tauW2_118[i],
        bw_method="scott"
    )
    kde28 = gaussian_kde(
        logDL_tauW2_28[i],
        bw_method="scott"
    )
    kdeWISE = gaussian_kde(
        logDL_tauW2_WISE[i],
        bw_method="scott"
    )
    
    kdeDL_tauW2.append(kde)
    kdeDL_tauW2_118.append(kde118)
    kdeDL_tauW2_28.append(kde28)
    kdeDL_tauW2_WISE.append(kdeWISE)

# Create a shared grid upon which to evaluate KDE probablity densities
kdeGrid = np.linspace(
    # Minimum and maximum distances from the H0 prior (in log10 space) with added padding (0.5)
    0.9670813842591461-0.5, 4.6055053736714875+0.5, 
    10000 # Grid contains 10000 cell resolution
)

# Evaluate the KDEs across the grid

# Lists holding the values of the KDE PDF at each grid cell
kdeDLVals_tauK = []
kdeDLVals_tauW1 = []
kdeDLVals_tauW1_118 = []
kdeDLVals_tauW1_28 = []
kdeDLVals_tauW1_WISE = []
kdeDLVals_tauW2 = []
kdeDLVals_tauW2_118 = []
kdeDLVals_tauW2_28 = []
kdeDLVals_tauW2_WISE = []

# K objects
for i in range(len(logDL_tauK)):
    kdeDLVals_tauK.append(kdeDL_tauK[i].evaluate(kdeGrid))

# W1 objects
for i in range(len(logDL_tauW1)):
    kdeDLVals_tauW1.append(kdeDL_tauW1[i].evaluate(kdeGrid))
    kdeDLVals_tauW1_118.append(kdeDL_tauW1_118[i].evaluate(kdeGrid))
    kdeDLVals_tauW1_28.append(kdeDL_tauW1_28[i].evaluate(kdeGrid))
    kdeDLVals_tauW1_WISE.append(kdeDL_tauW1_WISE[i].evaluate(kdeGrid))

# W2 objects
for i in range(len(logDL_tauW2)):
    kdeDLVals_tauW2.append(kdeDL_tauW2[i].evaluate(kdeGrid))
    kdeDLVals_tauW2_118.append(kdeDL_tauW2_118[i].evaluate(kdeGrid))
    kdeDLVals_tauW2_28.append(kdeDL_tauW2_28[i].evaluate(kdeGrid))
    kdeDLVals_tauW2_WISE.append(kdeDL_tauW2_WISE[i].evaluate(kdeGrid))
    
"""
Write the KDEs evaluated across the grid to .txt files for each sample

- Columns: Object
- Rows: KDE evaluated at one point on grid

"""

# K lags
with open("DistanceKDE_K.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauK)):
            if j-1 != len(kdeDLVals_tauK):
                writeVal += f"{kdeDLVals_tauK[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauK[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)

# W1 lags
with open("DistanceKDE_W1.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW1)):
            if j-1 != len(kdeDLVals_tauW1):
                writeVal += f"{kdeDLVals_tauW1[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW1[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)

with open("DistanceKDE_W1_118.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW1_118)):
            if j-1 != len(kdeDLVals_tauW1_118):
                writeVal += f"{kdeDLVals_tauW1_118[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW1_118[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)

with open("DistanceKDE_W1_28.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW1_28)):
            if j-1 != len(kdeDLVals_tauW1_28):
                writeVal += f"{kdeDLVals_tauW1_28[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW1_28[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)
            
with open("DistanceKDE_W1_WISE.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW1_WISE)):
            if j-1 != len(kdeDLVals_tauW1_WISE):
                writeVal += f"{kdeDLVals_tauW1_WISE[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW1_WISE[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)
            
# W2 lags
with open("DistanceKDE_W2.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW2)):
            if j-1 != len(kdeDLVals_tauW2):
                writeVal += f"{kdeDLVals_tauW2[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW2[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)

with open("DistanceKDE_W2_118.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW2_118)):
            if j-1 != len(kdeDLVals_tauW2_118):
                writeVal += f"{kdeDLVals_tauW2_118[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW2_118[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)

with open("DistanceKDE_W2_28.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW2_28)):
            if j-1 != len(kdeDLVals_tauW2_28):
                writeVal += f"{kdeDLVals_tauW2_28[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW2_28[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)
            
with open("DistanceKDE_W2_WISE.txt","w") as file:
    for i in range(len(kdeGrid)):
        writeVal = ""
        for j in range(len(kdeDLVals_tauW2_WISE)):
            if j-1 != len(kdeDLVals_tauW2_WISE):
                writeVal += f"{kdeDLVals_tauW2_WISE[j][i]}|"
            else:
                writeVal += f"{kdeDLVals_tauW2_WISE[j][i]}"
        if i != len(kdeGrid) - 1:
            file.write(writeVal + "\n")
        else:
            file.write(writeVal)



