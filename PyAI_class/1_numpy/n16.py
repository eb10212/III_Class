import  numpy as np

a = np.arange(6).reshape(3,2)
print(a)
print('')

b = np.arange(6).reshape(2,3)
print(b)
print('')

c = np.dot(a,b)  #矩陣內積
print(c)
print('')

d = np.matmul(a,b)  #矩陣內積
print(d)
print('')

print(c.T)  #矩陣轉置

print(a*b)  #矩陣相乘 ValueError: operands could not be broadcast together with shapes (3,2) (2,3) 無法使用廣播機制
