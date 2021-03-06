import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

N1 = int(1e3)  # Realizaciones
N2 = int(2)    # Modos
points = 1     # Puntos


P = 10
L = 100

delta_x = np.zeros(N1)


x_i = np.random.uniform(0, 1, 1)*L

for i in range(N1):
    omega_i = np.random.uniform(0, 1, N2)
    m = np.linspace(1, N2, N2)
    k = 2*m*np.pi/100
    theta = np.random.random(N2)*2.*np.pi
    delta_k = np.sqrt(P)*np.exp(1j*theta)
    
    delta_x[i] = (1/np.sqrt(L))*np.sum(2*np.real(delta_k*np.exp(1j*k*x_i)))

delta_x = np.array(sorted(delta_x))



y, x, z = plt.hist(delta_x, bins = 30, density = True, color = 'grey', edgecolor='grey',alpha = 0.5, label = r'$\delta(x)$')
# Altura histograma, ambos lados de las barras

bincenters = 0.5*(x[1:]+x[:-1])

density = np.array([])

for i in range(1, len(x)):
    inside = len(np.where((delta_x > x[i-1]) & (delta_x < x[i]))[0])
    density = np.append(density, inside)

def Gaussian(x, x0, sigma):
    return (1./np.sqrt(2.*np.pi*sigma**2))*np.exp(-(x-x0)**2/(2.*sigma**2))

sigma_x = np.std(delta_x)
mean_x = np.mean(delta_x)

Gauss = Gaussian(delta_x, mean_x, sigma_x)  

nonzero = np.where(density != 0)
error = np.zeros(len(y))
error[nonzero] = y[nonzero]/np.sqrt(density[nonzero])

plt.errorbar(bincenters, y, color = 'm', ls = '', yerr = error,capsize=2,fmt='.',ms=8,label = 'Errores',alpha=0.6)
plt.plot(delta_x, Gauss, 'k--',linewidth=2, label = 'Gauss')
plt.xlabel(r'$\delta(x)$', fontsize = 16)
plt.ylabel('P '+r'$(\delta(x))$', fontsize = 16)
plt.legend(fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('ap4_hist.png',dpi=300)
plt.show()


################################# 1000 puntos #################################

delta_x = np.zeros(N1)



omega_i = np.random.uniform(0, 1, N2)
m = np.linspace(1, N2, N2)
k = 2*m*np.pi/100
theta = np.random.random(N2)*2.*np.pi
delta_k = np.sqrt(P)*np.exp(1j*theta)

for i in range(N1):
    x_i = np.random.uniform(0, 1, 1)*L
    delta_x[i] = (1/np.sqrt(L))*np.sum(2*np.real(delta_k*np.exp(1j*k*x_i)))

delta_x = np.array(sorted(delta_x))

plt.figure()
y, x, z = plt.hist(delta_x, bins = 30, density = True, color = 'b', alpha = 0.5, label = r'$\delta(x)$')
# Altura histograma, ambos lados de las barras

bincenters = 0.5*(x[1:]+x[:-1])

density = np.array([])

for i in range(1, len(x)):
    inside = len(np.where((delta_x > x[i-1]) & (delta_x < x[i]))[0])
    density = np.append(density, inside)

def Gaussian(x, x0, sigma):
    return (1./np.sqrt(2.*np.pi*sigma**2))*np.exp(-(x-x0)**2/(2.*sigma**2))

sigma_x = np.std(delta_x)
mean_x = np.mean(delta_x)

Gauss = Gaussian(delta_x, mean_x, sigma_x)  

nonzero = np.where(density != 0)
error = np.zeros(len(y))
error[nonzero] = y[nonzero]/np.sqrt(density[nonzero])

plt.errorbar(bincenters, y, color = 'r', ls = '', yerr = error, label = 'Errores')
plt.grid()
plt.xlabel(r'$\delta(x)$', fontsize = 16)
plt.ylabel(r'$\mathcal{P}(\delta(x))$', fontsize = 16)
plt.title(r'$10^{3}$ puntos $x_{i}$', fontsize = 24)
plt.legend(loc = 'best', prop={'size':20})
