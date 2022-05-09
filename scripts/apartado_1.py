import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

N = int(1e5)
omega_i = np.random.uniform(0, 1, N)

P = 10
k = 2*np.pi/100
delta_i = np.array(sorted(np.sqrt(- P*np.log(omega_i))))


plt.figure(figsize=(6,4))
y, x, z = plt.hist(delta_i, bins = 30, density = True, color = 'grey',edgecolor='grey', alpha = 0.5, label = r'$|\delta_{k_{i}}|$')
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

plt.errorbar(bincenters, y, color = 'm', ls = '', yerr = error,capsize=2,fmt='.',ms=8,label = 'Errorbar',alpha=0.6)
plt.plot(delta_i, Ray, 'k--',linewidth=2,label ='Rayleigh')
plt.xlabel(r'$|\delta_{k}|$', fontsize = 16)
plt.ylabel('P '+r'$(|\delta_{k}|)$', fontsize = 16)
#plt.xlim((0,2.7))
plt.legend(fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('ap1_hist_5.png',dpi=300)
plt.show()
