# importing libraries #
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import sqrt, pi, exp, linspace, random
from scipy.optimize import curve_fit, leastsq
from lmfit import Model
from numpy.polynomial import polynomial as P
from numpy import array
import operator


# importing file #
readMiscut = pd.read_csv('data.csv', sep=',', skiprows=33)
# omega values
x = readMiscut.iloc[:,0].values
# phi values
y = readMiscut.iloc[:,1].values
# intensity values
z = readMiscut.iloc[:,2].values

# number of rows
nrolinhas = len(x)
# optimal omega value (max peak position)
omegaopt = [ ] 
# optimal phi value
phiopt = [ ]
# index
indice = 0


while indice < nrolinhas:
  # setting current phi value (in this file it goes from 0 to 345°)
  phiatual = y[indice]
  phiopt.append(float(phiatual))

  # variables for gaussian fit
  omega = [ ] 
  phi = [ ]
  intensidade = [ ]
  control = 0
  j = 0

  # filling the variables lists
  for control in range (0, nrolinhas-1):
    if y [control ] == phiatual:
      omega.append(x [control])
      phi.append(y[control])
      intensidade.append (z[control])
      j = j + 1

  # fitting Gaussian curve for one rocking curve peak
  n = len(omega) 
  i = 0
  soma = 0
  index, value = max(enumerate(intensidade), key=operator.itemgetter(1))
  mean = omega[index]
  sigma = 0
  for i in range (0,n-1):
    sigma = sigma + ((omega[i]-mean)**2)

  # guessing Gaussian curve sigma
  sigma = sqrt(sigma/n) 
  x0 = 0
  
  # defining Gaussian function
  def gauss (x, a, x0, sigma): 
    return a*exp(-(x-x0)**2/(2*sigma**2))

  popt, pcov = curve_fit(gauss, omega, intensidade, p0=[1500 ,mean, 0.15])
  # saving maximum omega value
  omegaopt.append(float(popt[1])) 
  indice = indice + j

  # plotting omega vs intensity graph (rocking curves for each phi value)
  plt.plot(omega, intensidade , 'b+:', label = 'data')
  plt.plot(omega, gauss(omega, *popt), 'r ', label = 'Gaussian fit')
  plt.legend()
  plt.xlabel('omega(°)')
  plt.ylabel('intensity (a.u.)')
  plt.show()
  
  
  # fitting sine curve

# number of data points
N = len(phiopt) 
t = np.linspace(0, 4*np.pi, N)

#guessing parameter
guess_freq = 0.5 
guess_amplitude = 3*np.std(omegaopt)/(2**0.5)
guess_phase = 0
guess_offset = np.mean(omegaopt)
p0=[guess_freq, guess_amplitude, guess_phase, guess_offset]

# sine function
def my_sin (k, freq, amplitude, phase, offset):
  return np.sin(k*freq + phase)*amplitude + offset

fit = curve_fit(my_sin, t, omegaopt, p0=p0)
data_first_guess = my_sin(t, *p0) 
# fitted sine
data_fit = my_sin(t, *fit[0]) 
poptb, pcovb = curve_fit(my_sin, t, omegaopt, p0=p0)

# print miscut angle value
print('MISCUT ANGLE (°) =' , poptb[1])

# plotting omega max vs phi curve
plt.plot(phiopt, omegaopt, '.', label = 'data') 
plt.plot(phiopt, data_fit, label = 'sine fit')
plt.xlabel('phi (°)')
plt.ylabel('omega (°)')
plt.legend()
plt.show()
