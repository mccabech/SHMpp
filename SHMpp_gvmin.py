import numpy as np
from math import sin, cos, pi, sqrt
from pylab import plot, show, xlabel, ylabel, legend
from numpy import zeros, empty, array, loadtxt, savetxt, copy

##########################################
#If you use this code please cite:

#SHM++: A Refinement of the Standard Halo Model for Dark Matter Searches
#Evans, McCabe, O'Hare
#arXiv:1809.?????

#The Earth's velocity for direct detection experiments 
#McCabe
#arXiv:1312.1355 
##########################################

##########################################
# Important physical constants
##########################################

#SHM++ Halo parameters
v0 = 233.
vesc = 580.
beta = 0.9
eta = 0.2

#Round component velocity dispersion 
sigR = v0/sqrt(2)
vdisR = [sigR, sigR, sigR]

#Sausage component velocity dispersion 
sigrSaus = sqrt( 3. / (2.*(3.-2.*beta)) ) * v0
sigtSaus = sqrt( 3.*(1.-beta) / (2.*(3.-2.*beta)) ) * v0
vdisSaus = [sigrSaus, sigtSaus, sigtSaus]

#Day=0 is midnight 1 Jan 2020
#Day=60.8 gives the average value of vE
Day = 60.8

print("Halo parameters:")
print("v0 is", v0)
print("vesc is", vesc)
print("beta is", beta)
print("eta is", eta)
print("Days after 1 Jan 2020:", Day)
print()

##########################################
# Parameters for the Monte Carlo integration
##########################################

#number of integration points in Monte Carlo
#higher = more accurate but slower
N = 1000000

#Importance sampling technique
#weight function is exp(-(x**2+y**2+z**2))
# draw random numbers from Normal distribution
mu, sigma = 0, 1/np.sqrt(2) 
x1 = np.random.normal(mu, sigma, N)
y1 = np.random.normal(mu, sigma, N)
z1 = np.random.normal(mu, sigma, N)


##########################################
# Calculate the Earths speed
# See arXiv:1312.1355 for derivation
##########################################

def vE(Day,v0):
    
    uSX = 11.1
    uSY = v0 + 12.24
    uSZ = 7.25

    n = Day + 7303.5 #Day=0 is midnight 1 Jan 2020
    T = (n)/36525.
    e1 = 0.9574 * pi/180.
    L = (280.460 + 0.9856474 * n) * pi/180.
    g = (357.528 + 0.9856003 * n) * pi/180.
    varpi = (282.932 + 0.0000471 * n) * pi/180.
    lam0 = varpi - 270.*pi/180.
    l = L + 2.*e1*sin(g) +5./4.*e1*e1*sin(2.*g)
    UEave = 29.79 #units are km/s

    bX = (5.536+0.013*T) * pi/180.
    bY = (-59.574+0.002*T) * pi/180.
    bZ = (-29.811+0.001*T) * pi/180.

    lamX = (266.840+1.397*T) * pi/180.
    lamY = (347.340+1.375*T) * pi/180.
    lamZ = (180.023+1.404*T) * pi/180.
    
    uEX = UEave * cos(bX) * ( sin(l-lamX) - e1 * cos(lamX-lam0) )
    uEY = UEave * cos(bY) * ( sin(l-lamY) - e1 * cos(lamY-lam0) )
    uEZ = UEave * cos(bZ) * ( sin(l-lamZ) - e1 * cos(lamZ-lam0) )

    vEx = uSX+uEX
    vEy = uSY+uEY
    vEz = uSZ+uEZ
    
    #units are km/s
    return [vEx,vEy,vEz]

##########################################
# Calculate the fR(v) normalisation factor
##########################################

#weight function is exp(-(x**2+y**2+z**2))
#introduced dimensionless velocities x y z
# x = vr / (sqrt2*sigr) etc
#weight function divides the velocity distribution
#integral performed in the galactic frame

vspeed2R = ( (sqrt(2)*vdisR[0]*x1)**2.
                + (sqrt(2)*vdisR[1]*y1)**2.
                + (sqrt(2)*vdisR[2]*z1)**2.
                )
                
fintegrandR = 1.
fconditionR = vesc**2 >= vspeed2R
sumNR = np.sum(fconditionR.astype(int))

Norm1R = ( sumNR * pi**(3./2.) * 2.**(3./2.)
              * vdisR[0] * vdisR[1] * vdisR[2] / N )

#Exact answer for Norm1R/v0**3 is known:
#it is 5.53407 for vesc = 580, v0=233
#print("Check numerical: round halo normalisation", Norm1R/v0**3.)


##########################################
# Calculate the fS(v) normalisation factor
#########################################

#same method, weight function etc as Norm1R

vspeed2S = ( (sqrt(2)*vdisSaus[0]*x1)**2.
                + (sqrt(2)*vdisSaus[1]*y1)**2.
                + (sqrt(2)*vdisSaus[2]*z1)**2.
                )
                
fintegrandS = 1.
fconditionS = vesc**2 >= vspeed2S
sumNS = np.sum(fconditionS.astype(int))

Norm1S = ( sumNS * pi**(3./2.) * 2.**(3./2.)
              * vdisSaus[0] * vdisSaus[1] * vdisSaus[2] / N )

#Exact answer for Norm1S/v0**3 is known:
#it is 2.13636 for vesc = 580, v0=233, beta=0.9
#print("Check numerical: Sausage halo normalisation", Norm1S/v0**3.)



##########################################
# Round component
# Calculate the g(vmin) integral for fR(v)
# output table of vmin [km/s] and g(vmin) [s/km]
#########################################

#weight function is exp(-(x2+y2+z2))
#integral performed in the galactic frame

print("Calculating gvmin for Round component")
print("vmin [km/s],","g(vmin) [s/km]")

ve = vE(Day,v0)

vspeed2R = ( (sqrt(2)*vdisR[0]*x1)**2.
                + (sqrt(2)*vdisR[1]*y1)**2.
                + (sqrt(2)*vdisR[2]*z1)**2.
                )
    
vspeed2bR = ( (sqrt(2)*vdisR[0]*x1-ve[0])**2.
                 + (sqrt(2)*vdisR[1]*y1-ve[1])**2.
                 + (sqrt(2)*vdisR[2]*z1-ve[2])**2.
                 )
    
fintegrandR = 1. / np.sqrt( vspeed2bR )

fcondition1R = vesc**2 >= vspeed2R
fresult1R = fcondition1R.astype(int)

Lvx = range(0,851,5)
LgvminR = []
for ivmin in Lvx:
    sat_all_cond = 0.
    vmin = ivmin * 1.
    #do the integral
    fcondition2R = vspeed2bR>=vmin**2
    fresult2R = fcondition2R.astype(int)
    sat_all_cond = fresult1R*fresult2R
    sumNR = np.sum(fintegrandR*sat_all_cond)
    
    int1R = ( sumNR * pi**(3./2.) * 2.**(3./2.)
             * vdisR[0] * vdisR[1] * vdisR[2] / N )

    gvminR = int1R / (Norm1R)

    LgvminR.append(gvminR)

    if ivmin % 100 == 0:
        print(ivmin,gvminR)


print("Export to gvmin_round.dat \n")
LoutR = list(zip(Lvx,LgvminR))
savetxt("gvmin_round.dat",LoutR)



##########################################
# Sausage component
# Calculate the g(vmin) integral for fS(v)
# output table of vmin [km/s] and g(vmin) [s/km]
#########################################

#weight function is exp(-(x2+y2+z2))
#integral performed in the galactic frame

print("Calculating gvmin for Sausage component")
print("vmin [km/s],","g(vmin) [s/km]")

ve = vE(Day,v0)

vspeed2S = ( (sqrt(2)*vdisSaus[0]*x1)**2.
                + (sqrt(2)*vdisSaus[1]*y1)**2.
                + (sqrt(2)*vdisSaus[2]*z1)**2.
                )
    
vspeed2bS = ( (sqrt(2)*vdisSaus[0]*x1-ve[0])**2.
                 + (sqrt(2)*vdisSaus[1]*y1-ve[1])**2.
                 + (sqrt(2)*vdisSaus[2]*z1-ve[2])**2.
                 )
    
fintegrandS = 1. / np.sqrt( vspeed2bS )

fcondition1S = vesc**2 >= vspeed2S
fresult1S = fcondition1S.astype(int)

Lvx = range(0,851,5)
LgvminS = []
for ivmin in Lvx:
    sat_all_cond = 0.
    vmin = ivmin * 1.
    #do the integral
    fcondition2S = vspeed2bS>=vmin**2
    fresult2S = fcondition2S.astype(int)
    sat_all_cond = fresult1S*fresult2S
    sumNS = np.sum(fintegrandS*sat_all_cond)
    
    int1S = ( sumNS * pi**(3./2.) * 2.**(3./2.)
             * vdisSaus[0] * vdisSaus[1] * vdisSaus[2] / N )

    gvminS = int1S / (Norm1S)

    LgvminS.append(gvminS)

    if ivmin % 100 == 0:
        print(ivmin,gvminS)

print("Export to gvmin_sausage.dat \n")
LoutS = list(zip(Lvx,LgvminS))
savetxt("gvmin_sausage.dat",LoutS)

##########################################
# Combine round and Sausage components for SHM++
# output table of vmin [km/s] and g(vmin) [s/km]
#########################################

print("Calculating gvmin for SHM++ ( with eta=",eta,")")
print("Export to gvmin_SHMpp.dat")
LgvminSHMpp = (1.-eta)*np.asarray(LgvminR) + eta*np.asarray(LgvminS)
LoutSHMpp = list(zip(Lvx,LgvminSHMpp))
savetxt("gvmin_SHMpp.dat",LoutSHMpp)

plot(Lvx, LgvminR,"r--", label='Round component')
plot(Lvx, LgvminS,"b--",  label='Sausage component')
plot(Lvx, LgvminSHMpp,"k", label='SHM++ with eta='+str(eta))
legend(loc='upper right')
xlabel("vmin [km/s]")
ylabel("g(vmin) [s/km]")
show()
