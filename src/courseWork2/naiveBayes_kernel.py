import cv2
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# image read
image_bgr = cv2.imread('data/imgs/0618.png')
image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

# mask read
mask = cv2.imread('data/imgs/mask0618.png',cv2.IMREAD_GRAYSCALE)
Y = mask.flatten()
Y = np.where(Y > 128, 1, 0) # 1 road, 0 not-road

# pixel to pixel_block
block_size = 5
def gaussian_kernel(size, sigma=1):
    """生成高斯滤波器"""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g / g.sum()
 
gaussian = gaussian_kernel(5, sigma=1)

kernels = [
    gaussian,  # 高斯滤波器
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),  # Sobel X
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)   # Sobel Y
]

# image size
height, width, _ = image_rgb.shape

features = []
labels = []

for i in range(0, height - block_size + 1, 1):
    for j in range(0, width - block_size + 1, 1):

        block = image_rgb[i:i+block_size, j:j+block_size]

        feature_vector = []

        # 应用卷积核到不同的颜色空间
        for color_space in ['RGB', 'LAB',]:
            if color_space == 'RGB':
                block_color = block
            # elif color_space == 'HSV':
            #     block_color = cv2.cvtColor(block, cv2.COLOR_RGB2HSV)
            elif color_space == 'LAB':
                block_color = cv2.cvtColor(block, cv2.COLOR_RGB2LAB)

            for kernel in kernels:
                for channel in range(block_color.shape[2]):
                    # 对每个通道应用卷积核
                    conv_result = cv2.filter2D(block_color[:,:,channel], -1, kernel)
                    # 将卷积结果的平均值添加到特征向量中
                    feature_vector.append(np.mean(conv_result))

        # extract labels
        block_mask = mask[i:i+block_size, j:j+block_size].flatten()
        # use center point
        label = 1 if mask[i+block_size//2,j+block_size//2] > 0.5 else 0
        # countPointvalues to set label
        # label = 1 if np.sum(block_mask) > (block_size * block_size / 2) else 0

        features.append(feature_vector)
        labels.append(label)


X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# model fit
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# cross verify
cv_scores = cross_val_score(gnb, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

y_pred = gnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 手动计算各项指标
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")

# use model to split the image
def segment_image(image, model, scaler, block_size=5):
    height, width, _ = image.shape
    segmented_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # Ensure that the border is not exceeded
            top = max(0, i - block_size // 2)
            bottom = min(height, i + block_size // 2 + 1)
            left = max(0, j - block_size // 2)
            right = min(width, j + block_size // 2 + 1)

            # extract data
            block = image[top:bottom, left:right]
            block = image_rgb[i:i+block_size, j:j+block_size]
            feature_vector = []
            for color_space in ['RGB', 'LAB',]:
                if color_space == 'RGB':
                    block_color = block
                # elif color_space == 'HSV':
                #     block_color = cv2.cvtColor(block, cv2.COLOR_RGB2HSV)
                elif color_space == 'LAB':
                    block_color = cv2.cvtColor(block, cv2.COLOR_RGB2LAB)

                for kernel in kernels:
                    for channel in range(block_color.shape[2]):
                        # 对每个通道应用卷积核
                        conv_result = cv2.filter2D(block_color[:,:,channel], -1, kernel)
                        # 将卷积结果的平均值添加到特征向量中
                        feature_vector.append(np.mean(conv_result))

            feature_vector = scaler.transform([feature_vector])
            # predict
            prediction = model.predict(feature_vector)
            # update split-image
            segmented_image[i, j] = prediction[0] * 255
    return segmented_image

# load test image
image_path = 'data/imgs/0618.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# split image
segmented_image = segment_image(image_rgb, gnb, scaler)
# image show
plt.imshow(segmented_image, cmap='gray')
plt.title("Segmented Image")
plt.show()

# after pre-process
def post_process(segmented_image, kernel_size=3, min_area=100):
    # Open operation removes small noise points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
    
    # Closed operation filling of small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Connected area analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    # null image
    processed_image = np.zeros_like(segmented_image, dtype=np.uint8)
    
    # Retain connected areas larger than min_area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            processed_image[labels == i] = 255
    
    return processed_image

processed_segmented_image = post_process(segmented_image)
# image show
plt.imshow(processed_segmented_image, cmap='gray')
plt.title("processed_segmented_image")
plt.show()