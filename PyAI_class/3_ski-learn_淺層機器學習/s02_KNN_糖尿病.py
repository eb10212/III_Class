import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split        #區分測試/訓練集
from sklearn.neighbors import KNeighborsClassifier
#交叉驗證
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#繪製學習曲線圖
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve         #外部程式碼(.py檔):有紅波浪正常

data = pd.read_csv('./datasets/diabetes.csv')               #原始資料集
print(data.shape)
print(data)
#print(data.columns)
#print(data.groupby('Outcome').size())

X = data.iloc[:, 0:8]  #八個特徵欄的資料(不含欄位8:結果)
Y = data.iloc[:, 8]    #結果欄的資料(欄位8)
print(X)
print(Y)
models = []  #準備放置多個候選模型(提供下列2種)
models.append(("KNN", KNeighborsClassifier(n_neighbors=5)))                                 #一般預設(查詢鄰居數)
models.append(("KNN-distance", KNeighborsClassifier(n_neighbors=3, weights="distance")))    #鄰居距離越遠權重weights越低(自動判斷)

#把train_test_split原始資料分成訓練與測試(80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#1.訓練模型-----------------------------------------80%
results = []    #2個模型->2種結果(放串列)
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
print(results)

print('======== Training model ============')
for i in range(len(results)):
    print(i)
    print("name: {}; score: {}".format(results[i][0],results[i][1]))
print('')
#模型"訓練"結果的預測度(分數)

#2."驗證"訓練模型
#模型驗證評估2種方式
results = []
for name, model in models:
    #方式1-顯示分數
    kfold = KFold(n_splits=10) # K折交叉驗證器，將資料折成10份(9份訓練, 1份測試)
    cv_result = cross_val_score(model, X, Y, cv=kfold) #交叉驗證評估分數
    results.append((name, cv_result))

    #方式2-畫出圖
    cv_ShuffleSplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #畫出學習曲線
    plt_learn = plot_learning_curve(model, "Learn Curve for KNN Diabetes",
                                    X, Y, ylim=(0., 1.2), cv=cv_ShuffleSplit)

print('======= Cross Validation ===========')
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(results[i][0],results[i][1].mean()))
print('')

#模型最終要看的是驗證後的分數,因為訓練模型有可能產生過你和的現象(分數很高,但實務上反而不能使用！！！)
#模型之後可以用下列方法預測未知的資料
#print("predict",models[0][1].predict(X),models[0][1].predict(X).shape)



#3.選取部分特徵-挑出兩個最佳特徵
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=2)
X_new = selector.fit_transform(X, Y)
print(X_new[0:5]) #列出出五筆->比對是哪兩欄的特徵被選出來-> 最相佳特徵分別為 Glucose（血糖濃度）和 BMI指數

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(
        results[i][0],results[i][1].mean()))

#繪製最佳特徵資料散布圖
plt.figure(figsize=(10, 6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='g', s=20, marker='o');  #陰性
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='r', s=20, marker='^');  #陽性

plt.show()







