from astroquery.ipac.ned import Ned
import time

objID = []
redshifts = []
zOG = []
RA = []
Dec = []
data = []
toggle = 1

if toggle == 0:
    with open("WISEDataOnly.txt", "r") as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            objID.append(lines[i][8:27])
            zOG.append(lines[i][28:33])
        
    for j in range(0, len(objID)):
        objID[j]="SDSS " + objID[j].strip()

    for obj in objID:
        try:
            result = Ned.query_object(obj)
            rshft = result["Redshift"][0]
            redshifts.append(rshft)
            time.sleep(1)
        except:
            redshifts.append(zOG[objID.index(obj)])

    for z in range(len(redshifts)):
        redshifts[z] = f"{redshifts[z]}\n"

    with open("WISERedshifts.txt", "w") as file:
        file.writelines(redshifts)

else:
    objID = ["Mrk 335","Mrk 590","IRAS 03450+0055","Akn 120","MCG+08-11-011","Mrk 79","Mrk 110","NGC 3227","NGC 3516","Mrk 744","NGC 4051","NGC 4151","NGC 4593","NGC 5548","Mrk 817","Mrk 509","NGC 7469"]
    for obj in objID:
        try:
            result = Ned.query_object(obj)
            rshft = result["Redshift"][0]
            ra = result["RA(deg)"][0]
            dec = result["DEC(deg)"][0]
            redshifts.append(rshft)
            RA.append(ra)
            Dec.append(dec)
            time.sleep(1)
        except:
            redshifts.append("none")
            RA.append("none")
            Dec.append("none")
    
    for i in range(len(redshifts)):
        data.append(f"{redshifts[i]}|{RA[i]}|{Dec[i]}\n")
    
    with open("KoshidaData.txt", "w") as file:
        file.writelines(data)
