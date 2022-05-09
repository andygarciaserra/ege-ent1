import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

N1 = int(1e2)
N2 = int(1e2)

P = 10
L = 100

mean_k2 = np.zeros(N1)

for i in range(N1):
    omega_i = np.random.uniform(0, 1, N2)
    m = np.linspace(1, N2, N2)
    k = 2*m*np.pi/100
    delta_k = np.sqrt(- P*np.log(omega_i))
    #delta_k = np.sqrt(P*np.full(100, 1))
    mean_k2[i] = (2/L)*np.sum(delta_k**2)  # <delta^2>

print('<<delta^2>>= '+str(np.mean(mean_k2)))
desvest_k2 = np.std(mean_k2)  # sigma(<delta^2>)
print('sigma(<delta^2>)= '+str(desvest_k2))

delta_k2 = mean_k2**2  # <delta^2>^2
mean_k2_real = np.mean(delta_k2)  # <<delta^2>^2>real
print('<<delta^2>^2>real= '+str(mean_k2_real))


mean_k2_real_2 = np.mean(mean_k2)  # <<delta^2>>real
print('<<delta^2>>real= '+str(mean_k2_real_2))


result = np.sqrt(np.abs(mean_k2_real - mean_k2_real_2**2))
print(result)
