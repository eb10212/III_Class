import numpy as np

a = np.array([[11,12,13],[23,24,25],[34,35,36]])  #二維陣列
print(a)
print('')

print(a[1])
print(a[1:])
print(a[1][1:])
print('')

print (a[...,1])  #...代表省略
print (a[1,...])
print (a[...,1:]) #第二列集剩下的所有元素

