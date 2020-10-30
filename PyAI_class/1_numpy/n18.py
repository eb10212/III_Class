import  numpy as np

a = np.arange(15).reshape(3,5)

print(a)
print('')

print(a.sum(axis=0))
print('')

print(a.sum(axis=1))
print('')

print(a.max(axis=0))
print('')

print(a.max(axis=1))
