# IMPORTING PACKAGES
import matplotlib.pyplot as plt
import numpy as np


# CODE
N = int(1e3)
omega_i = np.random.uniform(0, 1, N)

P = 10
k = 2*np.pi/100
delta_i = np.array(sorted(np.sqrt(- P*np.log(omega_i))))


y, x, z = plt.hist(delta_i, bins = 30, density = True, color = 'b', alpha = 0.5, label = r'$|\delta_{k}_{i}|$')
# Altura histograma, ambos lados de las barras

bincenters = 0.5*(x[1:]+x[:-1])

density = np.array([])

for i in range(1, len(x)):
    inside = len(np.where((delta_i > x[i-1]) & (delta_i < x[i]))[0])
    density = np.append(density, inside)

def Rayleigh(x, P):
    return 2*(x/P)*np.exp(-x**2/P)

Ray = Rayleigh(delta_i, P)  

nonzero = np.where(density != 0)
error = np.zeros(len(y))
error[nonzero] = y[nonzero]/np.sqrt(density[nonzero])

plt.errorbar(bincenters, y, color = 'r', ls = '', yerr = error, label = 'Errores')
plt.plot(delta_i, Ray, 'g', label = 'Distribución de Rayleigh')
plt.grid()
plt.xlabel(r'$|\delta_{k}|$', fontsize = 16)
plt.ylabel(r'$\mathcal{P}(|\delta_{k}|)$', fontsize = 16)
plt.title('Apartado 1', fontsize = 24)
plt.legend(loc = 'best', prop={'size':20})
plt.show()



