from astropy.table import Table

objID = []
trueID = []
z = []
lum = []
lumErr = []
data = []
found = False

with open("WISEDataOnly.txt", "r") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        objID.append(lines[i][9:27].strip())

tab = Table.read("dr7_bh_June_2010.fits")

for i in range(len(objID)):
    if "+" in objID[i]:
        if len(objID[i][:objID[i].index("+")])== 8:
            objID[i] = objID[i][:4] + "0" + objID[i][4:]
        if len(objID[i][objID[i].index("+")+1:]) == 7:
            objID[i] = objID[i][:objID[i].index("+")+5] + "0" + objID[i][objID[i].index("+")+5:]
    elif "-" in objID[i]:
        if len(objID[i][:objID[i].index("-")])== 8:
            objID[i] = objID[i][:4] + "0" + objID[i][4:]  
        if len(objID[i][objID[i].index("-")+1:]) == 7:
            objID[i] = objID[i][:objID[i].index("-")+5] + "0" + objID[i][objID[i].index("-")+5:]  

for j in range(len(objID)):
    found = False
    for i in range(len(tab)):
        if objID[j][:5]==tab["SDSS_NAME"][i][:5] and objID[j][10:len(objID[j])-3]==tab["SDSS_NAME"][i][10:len(tab["SDSS_NAME"][i])-3]:
            trueID.append(tab["SDSS_NAME"][i])
            z.append(tab["REDSHIFT"][i])
            if tab["REDSHIFT"][i] < 0.7:
                lum.append(tab["LOGL5100"][i])
                lumErr.append(tab["LOGL5100_ERR"][i])
            elif tab["REDSHIFT"][i] < 1.9:
                lum.append(tab["LOGL3000"][i])
                lumErr.append(tab["LOGL3000_ERR"][i])
            elif tab["REDSHIFT"][i] >= 1.9:
                lum.append(tab["LOGL1350"][i])
                lumErr.append(tab["LOGL1350_ERR"][i])
            found = True
            break   
    if found == False:
        trueID.append("none")
        z.append("none")
        lum.append("none")
        lumErr.append("none")


for i in range(len(lum)):
    data.append(f"{trueID[i]}|{z[i]}|{lum[i]}|{lumErr[i]}\n")

with open("WISEShenData.txt", "w") as file:
    file.writelines(data)