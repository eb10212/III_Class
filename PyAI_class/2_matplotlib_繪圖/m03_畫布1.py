
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi,np.pi,200)
y1 = np.cos(x)
y2 = np.sin(x)
y3 = np.tanh(x)

plt.figure(num='f1', figsize=(16,4))
plt.plot(x,y1)

plt.figure(num='f2', figsize=(16,4))
plt.plot(x,y2)

plt.figure(num='f1')
plt.plot(x,y2)  #將 y2 加入 f1

plt.figure(num='f1')
plt.plot(x,y3)  #將 y3 加入 f1
plt.show()