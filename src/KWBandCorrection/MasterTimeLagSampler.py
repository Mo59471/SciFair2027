#Master time lag MC sampler
#Samples from the time lag error margins to produce gammas and corrected time lags
#Produces 10,000, sampled, corrected time lags for each object

import random
import scipy
import statistics
import csv
import math

#Temporary list of reported redshifts from Mandal; should be CMB redshift
zCMB = [0.334,
0.080,
0.185,
0.118,
0.157,
0.056,
0.254,
0.302,
0.039,
0.065,
0.192,
0.051,
0.139,
0.086,
0.035,
0.329,
0.160,
0.082,
0.241,
0.054,
0.177,
0.079,
0.176,
0.045,
0.125,
0.152,
0.170,
0.094,
0.106,
0.140,
0.150,
0.230,
0.021,
0.050,
0.096,
0.093,
0.075,
0.295,
0.069,
0.085,
0.020,
0.077,
0.170,
0.023,
0.031,
0.282,
0.300,
0.154,
0.242,
0.180,
0.165,
0.055,
0.160,
0.034,
0.100,
0.110,
0.070,
0.075,
0.220,
0.050,
0.270,
0.060,
0.049,
0.288,
0.036,
0.054,
0.036,
0.100,
0.034,
0.030,
0.037,
0.067,
0.048,
0.085,
0.031,
0.070,
0.146,
0.206,
0.205,
0.106,
0.058,
0.320,
0.174,
0.736,
1.725,
0.805,
0.548,
1.431,
1.289,
1.181,
0.518,
1.119,
0.479,
0.569,
0.573,
0.854,
0.554,
0.734,
0.387,
0.378,
0.650,
0.681,
0.576,
0.632,
0.507,
0.949,
0.456,
0.748,
1.416,
1.028,
1.334,
0.371,
0.606,
1.033,
0.809,
0.782,
1.326,
1.030,
0.456,
0.719,
0.939,
0.694,
0.970,
0.998,
0.726,
0.682,
0.364,
1.402,
0.672,
0.818,
1.103,
0.453,
0.492,
0.675,
1.300,
0.821,
0.387,
1.262,
0.661,
0.615,
0.941,
0.565,
0.743,
0.934,
1.882,
0.367,
1.065,
1.211,
0.624,
0.735,
0.535,
0.796,
0.762,
1.171,
0.582,
0.907,
1.228,
0.513,
0.670,
0.352,
0.614,
0.840,
0.518,
0.719,
0.570,
0.589,
0.848,
1.228,
1.120,
0.848,
0.912,
0.500,
0.294,
0.239,
1.048,
1.085,
0.807,
0.486,
0.496,
0.560,
0.613,
1.459,
1.011,
0.343,
0.738,
0.814,
1.320,
0.708,
0.746,
1.020,
1.155,
0.394,
0.703,
0.745,
0.652,
0.738,
0.439,
0.776,
0.502,
0.644,
1.055,
0.621,
0.714,
0.375,
0.570,
1.398,
0.813,
0.550,
0.718,
0.535,
0.400,
0.329,
1.152,
0.906,
1.357,
1.023,
1.555,
0.733,
0.349,
0.620,
0.883,
0.771,
0.842,
0.927,
1.550,
0.950,
0.623,
0.711,
0.698,
1.389,
1.757,
0.361,
1.035,
0.734,
0.857,
0.710,
0.488,
1.155,
0.599,
0.610,
0.843,
1.065,
0.212,
0.883,
0.293,
1.048,
0.308,
0.566,
1.015,
0.489,
0.615,
0.770,
0.470,
0.422,
0.759,
1.076,
0.492,
0.866,
0.552,
0.526,
0.484,
0.988,
0.526,
1.688,
0.734,
1.379,
0.737,
0.861,
0.385,
0.453,
0.885,
1.083,
0.398,
0.554,
0.682,
0.768,
1.244,
0.849,
1.036,
0.334,
1.014,
0.804,
0.792,
0.855,
0.754,
0.600,
0.474,
0.574,
0.731,
0.741,
0.401,
0.600,
1.478,
1.183,
0.495,
1.020,
0.842,
1.112,
0.695,
0.442,
0.352,
1.087,
1.401,
1.056,
0.530,
0.318,
0.666,
0.878,
0.759,
0.776,
0.713,
0.822,
1.150,
0.606,
1.078,
0.668,
0.437,
1.138,
1.066,
0.488,
0.591,
0.829,
0.834,
0.676,
0.440,
0.902,
0.501,
0.362,
0.456,
0.420,
0.847,
0.582,
1.303,
0.771,
1.087,
0.786,
1.066,
0.960,
0.446,
0.704,
0.935,
0.588,
0.271,
0.971,
0.553,
0.852,
0.906,
0.569,
0.820,
0.806,
1.457,
0.752,
0.527,
0.525,
0.644,
1.304,
0.685,
1.727,
0.773,
0.567,
1.042,
0.615,
1.055,
0.432,
1.196,
0.502,
0.375,
1.199,
0.748,
0.633,
1.499,
0.719,
0.874,
0.600,
0.768,
0.344,
2.030,
0.716,
1.148,
0.374,
0.996,
0.745,
1.099,
1.858,
0.281,
0.704,
1.265,
0.495,
1.046,
0.794,
0.822,
0.542,
1.012,
0.750,
0.820,
2.001,
0.648,
0.889,
1.302,
1.564,
0.954,
1.029,
0.434,
0.920,
0.828,
0.633,
1.348,
0.585,
1.598,
0.546,
3.331,
0.471,
0.936,
0.569,
1.003,
0.654,
0.832,
0.960,
0.628,
0.756,
1.004,
0.904,
0.589,
1.499,
0.423,
0.444,
1.003,
1.345,
0.926,
1.278,
0.927,
0.879,
0.707,
0.420,
0.662,
0.598,
1.110,
0.705,
0.319,
1.346,
1.192,
0.516,
0.503,
0.312,
0.556,
0.524,
1.303,
1.083,
1.017,
1.085,
0.566,
0.404,
0.666,
0.718,
0.578,
0.714,
0.683,
0.720,
0.894,
0.427,
1.078,
1.034,
1.116,
0.982,
0.691,
0.362,
0.793,
1.006,
0.650,
0.485,
1.292,
0.422,
0.786,
1.089,
1.058,
0.561,
0.824,
1.605]
zFlowCorr = []
sTauW2 = []
sTauW1 = []
sTauW2Corr = []
sTauW1Corr = []
ratios = []
gammas = []
medGammaDistr = []

medianW2 = []
W2err = []
W2Err = []
modeW2 = []

medianW1 = []
W1err = []
W1Err= []
modeW1 = []

#Read W1 and W2 time lags into list from CSV
with open("RawWISELags.csv","r") as file:
    reader = csv.reader(file)
    data = list(reader)
    data = data[1:]

for i in data:
    if int(float(i[6])) != 0 and int(float(i[12])) != 0: #Check to see if there is a W1 AND W2 lag for an object
        medianW1.append(float(i[6]))
        W1err.append(float(i[7]))
        W1Err.append(float(i[8]))
        if float(i[7]) > float(i[8]):
            modeW1.append(float(i[6])-float(i[7]) * scipy.stats.norm.ppf((float(i[7])+float(i[8]))/(4*float(i[7])))) #Calculate mode from median based on skew
        elif float(i[7]) < float(i[8]):
            modeW1.append(float(i[6])-float(i[8])*scipy.stats.norm.ppf(1-((float(i[7])+float(i[8]))/(4*float(i[8])))))
        elif float(i[7]) == float(i[8]):
            modeW1.append(float(i[6]))         
            
        medianW2.append(float(i[12]))
        W2err.append(float(i[13]))
        W2Err.append(float(i[14]))
        if float(i[13]) > float(i[14]):     
            modeW2.append(float(i[12])-float(i[13]) * scipy.stats.norm.ppf((float(i[13])+float(i[14]))/(4*float(i[13]))))
        elif float(i[13]) < float(i[14]):
            modeW2.append(float(i[12])-float(i[14])*scipy.stats.norm.ppf(1-((float(i[13])+float(i[14]))/(4*float(i[14])))))
        elif float(i[13]) == float(i[14]):
            modeW2.append(float(i[12]))
            
    else: #Appends none if there is no value for one or more of the fields
        medianW1.append("none")
        W1err.append("none")
        W1Err.append("none")
        modeW1.append("none")
        medianW2.append("none")
        W2err.append("none")
        W2Err.append("none")
        modeW2.append("none")                

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
    for j in range(len(modeW1)):
        if modeW1[j] != "none":  
            #Sample W1 and W2 for each object j 
            W2 = sampleSplitNormal(modeW2[j], W2err[j], W2Err[j])
            W1 = sampleSplitNormal(modeW1[j],W1err[j],W1Err[j])
            while W2/W1 <= 0: #If less than 0, repeat sampling
                W2 = sampleSplitNormal(modeW2[j], W2err[j], W2Err[j])
                W1 = sampleSplitNormal(modeW1[j],W1err[j],W1Err[j])
            gam = math.log10(W2/W1)/math.log10(4.6/3.4) #Find the power laws corresponding to the given ratio; uses WISE nominal wavelengths W1 = 3.4 and W2 = 4.6
            
            #Append the raw sampled value to lists of lists (each sublist = distribution of sampled values for each object)
            if len(sTauW1)-1 !=j and len(sTauW2)-1 != j:
                sTauW1.append([W1])
                sTauW2.append([W2])
                ratios.append([W2/W1])
                gammas.append([gam])
                sTauW2Corr.append([(W2*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam)])
                sTauW1Corr.append([(W1*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam)])
            else:
                sTauW1[j].append(W1)
                sTauW2[j].append(W2)
                ratios[j].append(W2/W1)
                gammas[j].append(gam)
                sTauW2Corr[j].append((W2*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam))
                sTauW1Corr[j].append((W1*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam))                
        else:
            if len(sTauW1)-1 !=j and len(sTauW2)-1 != j:
                sTauW1.append(["none"])
                sTauW2.append(["none"])
                ratios.append(["none"])
                gammas.append(["none"]) 
                sTauW1Corr.append(["none"])
                sTauW2Corr.append(["none"])
    
#Calculate distribution of median gammas from which to sample from for objects without a W1 and W2 lag
for i in gammas:
    if i != "none":
        medGammaDistr.append(statistics.median(i))

#Sample from the distribution of median gammas and from the errors of each object to get corrected time lags for each object without a W1 and W2 lag 
for i in range(len(modeW1)):
    if sTauW1Corr[i][0] == "none" and modeW1[i] != "none":
        sTauW1Corr[i] = []
        for i in range(10000):
            sTauW1Corr[i].append((sampleSplitNormal(modeW1[i],W1err[i],W1Err[i])*((1+zCMB[i])**random.choice(medGammaDistr))*(2.2**random.choice(medGammaDistr)))/(3.4**random.choice(medGammaDistr)))
    if sTauW2Corr[i][0] == "none" and modeW2[i] != "none":
        sTauW2Corr[i] = []
        for i in range(10000):
            sTauW2Corr[i].append((sampleSplitNormal(modeW2[i],W2err[i],W2Err[i])*((1+zCMB[i])**random.choice(medGammaDistr))*(2.2**random.choice(medGammaDistr)))/(3.4**random.choice(medGammaDistr)))