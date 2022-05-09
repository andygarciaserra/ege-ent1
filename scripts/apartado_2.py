import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

N1 = int(1e3)
N2 = int(1e2)

P = 10
L = 100

omega_i = np.random.uniform(0, 1, N2)
m = np.linspace(1, N2, N2)
k = 2*m*np.pi/100
theta = np.random.random(N2)*2.*np.pi
delta_k = np.sqrt(- P*np.log(omega_i))*np.exp(1j*theta)

x_i = np.random.uniform(0, 1, N1)*L
delta_x = np.zeros(N1)

for i in range(N1):
    delta_x[i] = (1/np.sqrt(L))*np.sum(2*np.real(delta_k*np.exp(1j*k*x_i[i])))

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

print('sigma= '+str(sigma_x))

print('mean_x2= '+str(np.mean(delta_x**2)))
print('analytic_mean = '+str(2*(1/L)*np.sum(np.absolute(delta_k)**2)))


nonzero = np.where(density != 0)
error = np.zeros(len(y))
error[nonzero] = y[nonzero]/np.sqrt(density[nonzero])

plt.errorbar(bincenters, y, color = 'm', ls = '', yerr = error,capsize=2,fmt='.',ms=8,label = 'Errores',alpha=0.6)
plt.plot(delta_x, Gauss, 'k--',linewidth=2, label = 'Rayleigh')
plt.xlabel(r'$\delta(x)$', fontsize = 16)
plt.ylabel('P '+r'$(\delta(x))$', fontsize = 16)
plt.legend(fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('ap2_hist.png',dpi=300)
plt.show()

