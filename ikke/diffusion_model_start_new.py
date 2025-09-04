import numpy as np 
import matplotlib.pyplot as plt

print('a')
# First we import and plot the experimental data
D = np.loadtxt('metalblocks_lecture.txt', usecols = [0, 1, 2], unpack = True)

# Dimensionless time
t  = D[0,:]
T1 = D[1,:]
T2 = D[2,:]
T10 = np.mean(T1[70:91])
T20 = np.mean(T2[70:91])
DT0 = T20 - T10
Tmean0 = (T20 + T10) / 2

# Removing data before contact
mask = (T1 > T1[0])

t_masked = t[mask] - t[mask][0]
T1_rescaled = (2 * (T1 - Tmean0) / DT0)[mask]
T2_rescaled = (2 * (T2 - Tmean0) / DT0)[mask]

plt.figure(1)
plt.title("Experimental data")
# Plot rescaled temperature versus rescaled time
plt.plot(t_masked, T1_rescaled, color = 'k')
plt.plot(t_masked, T2_rescaled, color = 'r')
#plt.axis([0, 15, -1.0, 1.01])
plt.xlabel(r'$t$')
plt.ylabel(r'$2(T-<T_0>)/\Delta T_0$')
plt.legend(['Temperature bottom', 'Temperature top'])
plt.savefig("experimental_data.png")

# Then we try to model it
# Heat capacity ratio top to bottom
Ctb = 5

# Time scale (characteristic time)
tau = 200
# Number of heat packets
N = 800
# end of experiment
t_max = np.round(max(t_masked) / tau)
# Number of steps in simulation
nstep = int(t_max * N) 


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
  
        

# "time", in order to plot on same time scale
tstep = np.linspace(0, t_max, nstep) #)nstep / N

plt.figure(2)
plt.title("Algorithmic model")
# Experimental data
plt.plot(t_masked / tau, T1_rescaled, color = 'k')
plt.plot(t_masked / tau, T2_rescaled, color = 'r')
# Model data
plt.plot(tstep, Tt , color = 'r')
plt.plot(tstep, Tb, color = 'k')
plt.xlabel(r'$t / \tau $, steps')
plt.ylabel(r'$2(T-<T_0>)/\Delta T_0$')
plt.legend(['Temperature bottom', 'Temperature top'])
plt.savefig("alogrithmic_model.png")

