import os
import jieba
import pandas as pd

jieba.set_dictionary('./jieba_data/dict.txt.big')
stop_words_list = []
with open(file='jieba_data/movie_stop_word.txt',mode='r', encoding="UTF-8") as file:
    for line in file:
        line = line.strip()
        stop_words_list.append(line)
for i in range(1,6):
    dirpath = 'E:/專題-movie/moviestar{}'.format(i)
    f_list=os.listdir(dirpath)
# print(f_list)               #取出文章檔案list
    all_txt=[]
    for f_txt in f_list:
        with open('E:/專題-movie/moviestar{}/{}'.format(i,f_txt),'r',encoding='utf-8') as f:
            each_txt=f.read().replace('\n','').replace('-','').replace('_','').replace('~','')
            each_txt_jieba=jieba.lcut(each_txt,cut_all=False)       #精確模式
        for word in each_txt_jieba:
            if word not in stop_words_list:
                all_txt.append(word)

from collections import Counter
txt_count=Counter(all_txt)
# print(txt_count)
# print(txt_count.keys())


# # dataframe 處理數據
# seg_df = pd.DataFrame(all_txt, columns=['斷字'])
# seg_df['count'] = 1 #加上計數欄位
# seg_freq_df = seg_df.groupby('斷字').sum() #計算詞出現次數
# # print(seg_freq_df)



df=pd.DataFrame(columns=['詞','數'])
for i in txt_count:
    df.loc[i]=[i,txt_count[i]]
# print(df)
# df.to_csv('./123456.csv',index=False,encoding='utf-8-sig')

df2=pd.DataFrame(list(txt_count.items()),columns = ['key','value'])
# print(df2)
df2.to_csv('./word.csv',index=False,encoding='utf-8-sig')