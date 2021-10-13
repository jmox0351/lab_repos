from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sn
import numpy as np

def getData(l, i, tSteam):
    dataTemp = l[i]
    time = dataTemp[:,0]
    time = time - time[0]
    tem = (dataTemp[:,1] + dataTemp[:,2])/2
    thetaI = tSteam-tem[0]
    thetaT = tSteam-tem
    return(time, np.log(thetaI/thetaT))

#import first file as df then convert to numpy array
data1 = pd.read_csv("run1groupD.txt", sep = '\t')
data1 = data1.drop('Unnamed: 4', axis = 1)
data1 = data1.apply(pd.to_numeric)
data1 = data1.to_numpy()
l = [data1]

#import the rest of the files
for i in range(2, 17):
    docString = "run" + str(i) + "groupD.txt"
    dataTemp = pd.read_csv(docString, sep = '\t')
    dataTemp = dataTemp.drop('Unnamed: 4', axis = 1)
    dataTemp = dataTemp.apply(pd.to_numeric)
    dataTemp = dataTemp.to_numpy()
    l.append(dataTemp)

#trim the arrays so that the average of the center and edge are less than 30 or greater than 70
for i in range(0,16):
    tempAr = l[i]
    tempAr = tempAr[np.logical_and((tempAr[:,1]+tempAr[:,2])/2 >= 30, (tempAr[:,1]+tempAr[:,2])/2 <= 70)]
    l[i] = tempAr

#fit a linear curve for each run for average temp vs time
Dt = np.array([])
DTheta = np.array([])
tSteam = 107

for i in range(0,16):
    time, theta = getData(l, i, tSteam)
    Dt = np.append(Dt,time[-1])
    DTheta = np.append(DTheta,theta[-1])

m = 59.2 #kg of water in vessel
cp = 4181.5 #cp at 1 atm, 50 degree C
As = 0.5868 #surface area of inside
u = DTheta*m*cp/(As*Dt)

print('average U for 250 rpm', sum([u[0],u[4],u[8],u[12]])/4)
print('average U for 500 rpm', sum([u[1],u[5],u[9],u[13]])/4)
print('average U for 750 rpm', sum([u[2],u[6],u[10],u[14]])/4)
print('average U for 1000 rpm', sum([u[3],u[7],u[11],u[14]])/4)

fig, axs = plt.subplots(2,2, figsize=(18,15))
tSteam = 107 #engineering toolbox, boiling point at 19 psia

for i in range(0,16):
    timeTemp, temTemp = getData(l, i, tSteam)
    if (i % 4 == 0):
        axs[0,0].plot(timeTemp, temTemp)
    elif (i % 4 == 1):
        axs[0,1].plot(timeTemp, temTemp)
    elif (i % 4 == 2):
        axs[1,0].plot(timeTemp, temTemp)
    else:
        axs[1,1].plot(timeTemp, temTemp)

axs[0,0].set_title('250 RPM', fontsize = 18)
axs[0,1].set_title('500 RPM', fontsize = 18)
axs[1,0].set_title('750 RPM', fontsize = 18)
axs[1,1].set_title('1000 RPM', fontsize = 18)

for i in range(0,2):
    for j in range(0,2):
        axs[i,j].set_xlabel('Time (s)', fontsize = 16)
        axs[i,j].set_ylabel('Temperature (C)', fontsize = 16)
plt.show()

ds1 = np.loadtxt('run1SugGroupD.txt', skiprows = 1)
ds1 = ds1[0:1000,:]
s = [ds1]

for i in range (2,9):
    docString = "run" + str(i) + "SugGroupD.txt"
    ds = np.loadtxt(docString, skiprows = 1)
    s.append(ds)

for i in range(0,8):
    tempAr = s[i]
    tempAr = tempAr[np.logical_and((tempAr[:,1]+tempAr[:,2])/2 >= 30, (tempAr[:,1]+tempAr[:,2])/2 <= 70)]
    s[i] = tempAr

Dtsug = np.array([])
DThetasug = np.array([])
tSteam = 107

for i in range(0,8):
    time, theta = getData(s, i, tSteam)
    Dtsug = np.append(Dtsug,time[-1])
    DThetasug = np.append(DThetasug,theta[-1])

specificGrav = np.array([1.02, 1.035, 1.02, 1.083, 1.04, 1.02, 1.061, 1.04])
m = 59.2 #kg of water in vessel
cp = np.array([4080.0, 3990.0, 4080.0, 3760.0, 3970.0, 4080.0, 3870.0, 3970.0]) #cp at 1 atm, 50 degree C
As = 0.5868 #surface area of inside
usug = DThetasug*m*specificGrav*cp/(As*Dtsug)

con = np.array([0.05, 0.09, 0.05, 0.2, 0.1, 0.05, 0.15, 0.1])
con = con*100
con = sm.add_constant(con)
model = sm.OLS(usug, con)
results = model.fit()
results.params

con = np.array([0.05, 0.09, 0.05, 0.2, 0.1, 0.05, 0.15, 0.1])
con = con*100
plt.figure(figsize = (12,12))
sn.regplot(x = con, y = usug, ci = 68)

plt.legend(labels=['y = -2.901*x + 1064.3'], fontsize = 14)
plt.xlabel('Sugar concentration (weight percent)', fontsize = 16)
plt.ylabel('Overall heat transfer coefficent W/m^K', fontsize = 16)
plt.show()