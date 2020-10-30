import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi,np.pi,200)
y1 = np.cos(x)
y2 = np.sin(x)
y3 = np.tanh(x)
y4 = np.exp(x)

plt.figure(figsize=(18,8))

plt.subplot(2,2,1)
plt.plot(x,y1)

plt.subplot(2,2,2)
plt.plot(x,y2)

plt.subplot(2,2,3)
plt.plot(x,y3)

plt.subplot(2,2,4)
plt.plot(x,y4)

plt.show()



