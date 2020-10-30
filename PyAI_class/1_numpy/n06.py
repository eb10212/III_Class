import numpy as np

a = np.linspace(10, 20, 5)
print(a)
print('')
a = np.linspace(10, 20, 5, endpoint = False) #切5+1等份最後一個等份不要
print(a)

a = np.linspace(10, 20, 5, endpoint = True)
print(a)
print('')

b = np.linspace(0, 2, 9, endpoint = False).reshape(3,3)
print(b)


c = np.logspace(0, 9, 10, base=2, dtype='i4').reshape(2,5)
print (c)