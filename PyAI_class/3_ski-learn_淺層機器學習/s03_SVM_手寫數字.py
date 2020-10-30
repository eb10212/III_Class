import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

#針對資料集的內容--------------------------------------------------
#取前八筆，顯示標記與圖片
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(8, 6))
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest') #內插法
    plt.title('Digit: %i' % label, fontsize=20);

#畫出第一筆數字圖片
plt.figure()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest') #內插法
#print("shape of raw image data: {0}".format(digits.images.shape))
#print(digits.data)
print("shape of data: {0}".format(digits.data.shape)) #64個特徵

#模型訓練--------------------------------------------------
#區分80%訓練資料與20%測試資料
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)

#建立 SVM 模型
from sklearn import svm
model = svm.SVC(gamma=0.001, C=100., probability=True) #C-Support Vector Classification.
model.fit(Xtrain, Ytrain) #訓練模型
model.score(Xtest, Ytest) #模型評分

#繪製學習曲線圖
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt_learn = plot_learning_curve(model, "Learn Curve for SVM ", digits.data, digits.target, ylim=(0., 1.2), cv=cv)

#精確度(使用20%的測試樣本在跑自己訓練出來的模型看準不準)
from sklearn.metrics import accuracy_score
Ypred = model.predict(Xtest)
print("accuracy_score",accuracy_score(Ytest, Ypred))

#完整模型評估報告
from sklearn.metrics import classification_report
print(classification_report(Ytest, Ypred))

#顯示預測狀況圖片(取4*4=16 張範本)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32,
            transform=ax.transAxes,
            color='green' if Ypred[i] == Ytest[i] else 'red')       #若預測值=實際值:綠色
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32,
            transform=ax.transAxes,
            color='black')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()