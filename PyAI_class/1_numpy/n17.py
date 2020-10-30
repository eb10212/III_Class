import  numpy as np

a = np.arange(6)
b = np.array([0,1,2,3,4])
c = np.array([6,1,1,1,1])
print(a)
print('')

print(a.sum())
print('')

print(a.mean(),b.mean(),c.mean())
print('')

print(a.std(),b.std(),c.std())
print('')

print(a.min())
print('')

print(a.max())
print('')
print('........................................')
a = np.random.randint(1,10,6)

print(a)
print('')

print(a.argmin())
print('')

print(a.argmax())
print('')

print(np.cos([0, np.pi,  2*np.pi]))
print('')

print(np.exp([1,2,3]))   #無理數e也稱為歐拉數，約為2.718281，e的x次方
print('')

print(np.sqrt([1,4,9,16]))
print('')