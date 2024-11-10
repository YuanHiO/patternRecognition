import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

image = cv2.imread('../data/imgs/1066.png')
# 转换为RGB格式（opencv读取为BGR）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 加载生成的掩码
mask = cv2.imread('../data/imgs/mask1066.png', cv2.IMREAD_GRAYSCALE)
Y = mask.flatten()
Y = np.where(Y > 128, 1, 0)  # 1为道路，0为非道路

# 输出分布情况
series = pd.Series(Y)
value_counts = series.value_counts()
print(value_counts)

# 特征提取
# 获取图像的尺寸
h, w, _ = image_rgb.shape
# 转换到HSV颜色空间
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
# 转换到灰度图像
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
# 计算灰度共生矩阵 (GLCM)
glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                    symmetric=True, normed=True)
# 提取纹理特征
# 对比度：图像中局部灰度差异的大小，反映了图像的清晰度和纹理的粗细
contrast = graycoprops(glcm, 'contrast')
# 不相似性： 图像中灰度级之间的不相似程度。
dissimilarity = graycoprops(glcm, 'dissimilarity')
# 同质性：图像中灰度级分布的均匀性，反映了图像的平滑程度。
homogeneity = graycoprops(glcm, 'homogeneity')
# 能量： 描述图像中灰度级分布的集中程度，反映了图像的均匀性和纹理的一致性。
energy = graycoprops(glcm, 'energy')
# 相关性： 描述图像中灰度级之间的线性相关性，反映了图像的纹理方向性
correlation = graycoprops(glcm, 'correlation')
# 角二阶矩：图像中灰度级分布的集中程度，反映了图像的均匀性和纹理的一致性。 
ASM = graycoprops(glcm, 'ASM')

# 构建特征向量 (R, G, B, H, S, V, x, y, contrast, dissimilarity, homogeneity, energy, correlation, ASM)
X = []
for i in range(h):
    for j in range(w):
        pixel = image_rgb[i, j]
        hsv_pixel = image_hsv[i, j]
        # 获取当前像素点的纹理特征
        texture_features = [
            contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0],
            energy[0, 0], correlation[0, 0], ASM[0, 0]
        ]
        X.append([
            pixel[0], pixel[1], pixel[2],
            hsv_pixel[0], hsv_pixel[1], hsv_pixel[2],
            # i, j,
            *texture_features
        ])

# 将 X 转换为 numpy 数组
X = np.array(X)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# 归一化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 创建SVM分类器
clf = svm.SVC(kernel='rbf') 
# 训练SVM分类器
clf.fit(X_train, Y_train)
# 在测试集上评估模型
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))

# 生成预测结果的图像
predicted_mask = np.zeros_like(mask)
for i in range(h):
    for j in range(w):
        pixel = image_rgb[i, j]
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2HSV)[0, 0]
        texture_features = [
            contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0],
            energy[0, 0], correlation[0, 0], ASM[0, 0]
        ]
        feature_vector = np.array([
            pixel[0], pixel[1], pixel[2],
            hsv_pixel[0], hsv_pixel[1], hsv_pixel[2],
            # i, j,
            *texture_features
        ]).reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)
        predicted_mask[i, j] = clf.predict(feature_vector)[0]

# 显示预测结果的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Predicted Mask')
plt.imshow(predicted_mask, cmap='gray')
plt.axis('off')

plt.show()