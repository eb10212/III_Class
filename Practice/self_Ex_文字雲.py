import jieba
jieba.set_dictionary('./jieba_data/dict.txt.big')
stop_words_list = []
with open(file='jieba_data/stop_words.txt',mode='r', encoding="UTF-8") as file:
    for line in file:
        line = line.strip()
        stop_words_list.append(line)


with open('./moviestar1/92_01.txt','r',encoding='utf-8') as f:
    a=f.read()
    # print(a)

b=jieba.lcut(a)
c=[]
for i in b:
    if i not in stop_words_list:
        c.append(i)
# print(c)
from collections import Counter
d=Counter(c)
# print(d)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams['font.sans-serif']=['SimHei']
#用來顯示負號
plt.rcParams['axes.unicode_minus']=False
wordcloud = WordCloud(font_path='fonts/TaipeiSansTCBeta-Regular.ttf').generate_from_frequencies(d)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
