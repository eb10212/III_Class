import numpy as np

a= np.zeros([2,3])
print(a)

b= np.ones([2,3])
print(b)

c = np.array([[1,2,3],[4,5,6]])

c0 = np.zeros_like(c)
print(c0)
c1 = np.ones_like(c)
print(c1)

d = np.random.random((2,3))
print(d)

f = np.eye(3, dtype=int )
print(f)

f1 = np.eye(3, k=1)
print(f1)

f1 = np.eye(3, k=-2)
print(f1)

