import cv2 as cv
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from skimage.feature import hog
from joblib import dump, load

# 加载数据，这里使用 sklearn 自带的手写数字数据集
digits = load_digits()
X, y = digits.images, digits.target

# 提取 HOG 特征
X_hog = []
for img in X:
    img_uint8 = (img * 16).astype(np.uint8)
    hog = cv.HOGDescriptor((8, 8), (4, 4), (4, 4), (4, 4), 9)
    fd = hog.compute(img_uint8)
    X_hog.append(fd.flatten())
X_hog = np.array(X_hog)

#===== 使用 scikit‐learn 的 SVM 实现 =====
# 训练 ——同时保留原始图像用于可视化
X_train, X_test, imgs_train, imgs_test, y_train, y_test = train_test_split(X_hog, X, y, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
print(f'准确率: {clf.score(X_test, y_test):.2%}')

# 用原始图像可视化部分测试样本及其预测结果
for i in range(5):
    # imgs_test 存储原始 8x8 浮点图像（0..16），先还原为 uint8 后放大显示
    img = (imgs_test[i] * 16).astype(np.uint8)
    img = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)
    pred = clf.predict([X_test[i]])[0]
    true = y_test[i]
    # 图中添加预测结果文字
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.putText(img, f'True: {true}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 0)
    cv.putText(img, f'sklearn Pred: {pred}', (10, 70), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv.imshow(f'sklearn SVM‐ True: {true} Pred: {pred}', img)
    cv.waitKey(0)
cv.destroyAllWindows()

# 保存模型
dump(clf, 'hog_svm.joblib')

# ===== 使用 opencv‐python 的 SVM 实现（与 scikit‐learn 的实现类似）
# 准备数据（OpenCV 要求 float32 的样本矩阵和 int32 的响应）
X_train_cv = X_train.astype(np.float32)
X_test_cv = X_test.astype(np.float32)
y_train_cv = y_train.astype(np.int32)
y_test_cv = y_test.astype(np.int32)

svm_cv = cv.ml.SVM_create()
svm_cv.setType(cv.ml.SVM_C_SVC)
svm_cv.setKernel(cv.ml.SVM_LINEAR)
svm_cv.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

# 训练
train_data = cv.ml.TrainData_create(X_train_cv, cv.ml.ROW_SAMPLE, y_train_cv)
svm_cv.train(train_data)

# 评估准确率
_, resp = svm_cv.predict(X_test_cv)
preds_cv = resp.flatten().astype(np.int32)
acc_cv = (preds_cv == y_test_cv).mean()
print(f'OpenCV SVM 准确率: {acc_cv:.2%}')

# 可视化部分测试样本的预测（来自 OpenCV SVM）
for i in range(5):
    img = (imgs_test[i] * 16).astype(np.uint8)
    img = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    pred_opencv = int(preds_cv[i])
    true = int(y_test_cv[i])
    cv.putText(img, f'True: {true}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 0)
    cv.putText(img, f'OpenCV Pred: {pred_opencv}', (10, 70), cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255,2),2)
    cv.imshow(f'OpenCV SVM‐ True:{true} Pred:{pred_opencv}', img)
    key = cv.waitKey(0)
    if key == 27: # 按 Esc 可提前退出
        break
cv.destroyAllWindows()

# 保存 OpenCV SVM 模型
svm_cv.save('hog_svm_opencv.yml')