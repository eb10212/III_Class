import pandas as pd
with open('./test1.csv','r',encoding='utf-8')as f:
    a=f.read()
a=a.split('\n')
# print(a)
# print(a[0])
# print(type(a[0]))
# print(len(a))
b=[i.split(',') for i in a]
# print(b)
column=b[0]
df3=pd.DataFrame(data=b[1:],columns=column)
# print(df3)

with open('./test1.csv','r',encoding='utf-8')as f:
    c=f.read()
c=c.split('\n')
d=[c.split(',') for c in c]
column=d[0]
df4=pd.DataFrame(data=d[1:],columns=column)
# print(df4)


lis=[]
print(b[1]+d[1])
for i in range(len(b)):
    lis+=[b[i]+d[i]]
print(lis)
df5=pd.DataFrame(data=lis[1:],columns=lis[0])
print(df5)