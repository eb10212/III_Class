# NumpPy 三維陣列的索引
import numpy as np

a = np.arange(16).reshape(2,2,4)

print(a)
print('')

print(a[0])  #切片擷取
print('')

print(a[ 0 , : , 1:3 ])
print('')

print(a[ 0 , 1 , :-1 ])
print('')

print(a[0,1]) #陣列索引
print('')

print(a[0][1]) #切片擷取
print('')

print(a[0,1,-2:])  #陣列索引
print('')

print(a[0][1][-2:]) #切片擷取
print('')

