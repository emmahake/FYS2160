import numpy as np #
import matplotlib.pyplot as plt 

D = np.loadtxt('metalblocks_lecture.txt', usecols = [0, 1, 2], unpack = True)
# Guessing the characteristic time
# Dimensionless time
t  = D[0,:]
T1 = D[1,:]
T2 = D[2,:]
T10 = np.mean(T1[70:91])
T20 = np.mean(T2[70:91])
DT0 = T20 - T10
Tmean0 = (T20 + T10) / 2

# Heat capacity ratio top to bottom
Ctb = 2.3
# Time scale (characteristic time)
tau = 100
# Number of heat packets
N = 80000 #800
# end of experiment
t_max = np.round(max(t) / tau)
# Number of steps in simulation
nstep = int(t_max * N) 

plt.figure(1)
# Plot rescaled temperature versus rescaled time
plt.plot(t / tau, 2 * (T1 - Tmean0) / DT0, color = 'k')
plt.plot(t / tau, 2 * (T2 - Tmean0) / DT0, color = 'r')
plt.axis([0, 15, -1.0, 1.01])
plt.xlabel(r'$t/\tau$')
plt.ylabel(r'$2(T-<T_0>)/\Delta T_0$')
plt.legend(['Temperature top','Temperature bottom'])
plt.savefig("fig1.png")


# Initialize temperature-time arrays
Tt = np.zeros(nstep, float)
Tb = np.zeros(nstep, float)
# Initial temperature top
Tt[0] = 1
# Initial temperature bottom
Tb[0] = -1
# Room temperature
Tr = -1

for i in range(1, nstep):
    # Random number between 2 and -2
    r = 4 * np.random.rand(1, 1) - 2 
    # Temperature difference top to bottom
    DT = Tt[i - 1] - Tb[i - 1]
    if r < DT:
        # Move heat quanta from top to bottom
        Tt[i] = Tt[i - 1] - 1 / N 
        Tb[i] = Tb[i - 1] + Ctb / N
    else:
        # Move heat quanta from bottom to top
        Tt[i] = Tt[i - 1] + 1 / N
        Tb[i] = Tb[i - 1] - Ctb / N
        
plt.figure(2)
plt.plot(range(0, nstep), Tt, color = 'r')
plt.plot(range(0, nstep), Tb, color = 'k')
plt.xlabel('steps')
plt.ylabel(r'$2(T-<T_0>)/\Delta T_0$')
plt.legend(['Temperature top','Temperature bottom'])
plt.savefig("fig2.png")


