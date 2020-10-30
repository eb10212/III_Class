#pip install pyhanlp        #自然語言處理包-中文分詞/詞性
from pyhanlp import *

import jieba
import jieba.posseg as pseg         #斷詞性
jieba.set_dictionary('./jieba_data/dict.txt.big')       #指定繁體字的字典-老師

#jieba斷字
text_str = '今天天氣真好'
seg_result = jieba.lcut(text_str, cut_all=False)         #精確模式斷詞(cut_all=False)
seg_result = jieba.lcut(text_str, cut_all=True)          #全模式斷詞(cut_all=True)
seg_result = jieba.lcut_for_search(text_str, HMM=True)   #搜尋模式斷詞(HMM=True)
seg_result = jieba.lcut(text_str, use_paddle=True)       #Paddle模式斷詞(use_paddle=True)

# print(' / '.join(seg_result))                          #lcut:斷字後,以list呈現

# jieba.enable_paddle()
# seg_result = pseg.lcut(text_str, use_paddle=True)
# for w, p in seg_result:
#     print("%s, %s"%(w, p))

#pyhanlp斷字斷詞
text_str_2 = '皇家鹽湖城梅西煤球王c羅費城聯合'
# print(HanLP.segment(text_str))



#使用停用字典
text_str_3 = '我是一位小學生，從小學習鋼琴，希望成為youtuber'
seg_result = jieba.lcut(text_str_3, cut_all=False)
# print('原始斷詞:',seg_result)

stop_words_list = []
with open(file='jieba_data/stop_words.txt',mode='r', encoding="UTF-8") as file:
    for line in file:
        line = line.strip()
        stop_words_list.append(line)

seg_result_stopword = []
for term in seg_result:
    if term not in stop_words_list:
        seg_result_stopword.append(term)
# print('使用停用字典:',seg_result_stopword)        #例如:是/"，"就不會出現


#練習
fileAllLines = []
with open('./moviestar1/92_01.txt','r',encoding='utf-8') as f:
    for line in f:
        fileAllLines.append(line)
print(fileAllLines)
seg = []
for i in range(len(fileAllLines)):
    cut_list = jieba.lcut(fileAllLines[i], cut_all=False)
    [ [item] for item in cut_list if len(item) > 1 ]
    seg.append([' '.join([ item for item in cut_list if len(item) > 1 ])])
seg = [ s for s in seg if len(s) > 0]
print(seg)

