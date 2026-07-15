objID = []
trueID = []
lum = []
lumErr = []
data = []
z = []
found = True

#Names: 7:25
with open("WISEDataOnly.txt", "r") as file:
    lines = file.readlines()
    for i in range(1, 85):
        objID.append(lines[i][9:27].strip())
        
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
            
with open("LiuNames.txt", "r") as file:
    lines = file.readlines()
    for j in range(len(objID)):
        found = False
        for i in range(len(lines)):
            if objID[j][:5]==lines[i][7:25][:5] and objID[j][10:len(objID[j])-3]==lines[i][7:25][10:len(lines[i][7:25])-3]:
                trueID.append(lines[i][7:25].strip())
                z.append(lines[i][54:64].strip())
                
                with open("LiuL5100.txt", "r") as file2:
                    lines2 = file2.readlines()
                    lum.append(lines2[i][1360:1378])
                    lumErr.append(lines2[i][1379:1403])
                found = True
                break
               
        if found == False:
            trueID.append("none")
            z.append("none")
            lum.append("none")
            lumErr.append("none")


for i in range(len(lum)):
    data.append(f"{trueID[i]}|{z[i]}|{lum[i]}|{lumErr[i]}\n")

with open("WISELiuData.txt", "w") as file:
    file.writelines(data)
     