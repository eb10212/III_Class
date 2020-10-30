import numpy as np
#陣列切片/取值/設定值 (一維陣列)
a = np.arange(6)
print(a) #[0 1 2 3 4 5]

print(a[2])
print(a[3:5])#[3 4]
print(a[2:-1])#[2 3 4]
print(a[::1])#[0 1 2 3 4 5]
print(a[::-1])#[5 4 3 2 1 0]
print(a[::2])#0 2 4]
print(a[::-2])#[5 3 1]


a[2] = 0
print(a)#[0 1 0 3 4 5]

a[3:5] = 0
print(a)#[0 1 0 0 0 5]
print(a[3:5])#[0 0]
print(a[1:])#[1 0 0 0 5]
