import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np

x = np.linspace(-np.pi,np.pi,200)
y1 = np.cos(x)
y2 = np.sin(x)
y3 = np.tanh(x)
y4 = np.exp(x)

plt.figure(figsize=(18,4))

G = grd.GridSpec(2,3)

axes1 = plt.subplot(G[0,:])
plt.plot(x,y1)

axes2 = plt.subplot(G[1,0])
plt.plot(x,y2)

axes2 = plt.subplot(G[1,1])
plt.plot(x,y3)

axes2 = plt.subplot(G[1,2])
plt.plot(x,y4)

plt.show()


