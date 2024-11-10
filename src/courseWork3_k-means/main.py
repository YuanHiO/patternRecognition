import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score

import time
# data process
def loadImage(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    return image

def process_image(image):
    data = image.reshape((-1,3))
    return data

# k-means
def init_centroids(data,K):
    np.random.seed(42)
    indices = np.random.choice(len(data), K, replace=False)
    centroids = data[indices]
    return centroids

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:,np.newaxis] - centroids,axis=2)
    clusterLabels = np.argmin(distances,axis=1)
    return clusterLabels

def update_centroids(data, clusterLabel, k):
    newCentroid = np.array([data[clusterLabel == i].mean(axis=0) for i in range(k)])
    return newCentroid

def isConverged(oldCentroids, centroids, tolerence = 1e-5):
    return np.linalg.norm(centroids - oldCentroids) < tolerence

def k_means(data, k, iter=100):

    start = time.time()

    centroids = init_centroids(data, k)
    for i in range(iter):
        labels = assign_clusters(data,centroids)
        newCentroids = update_centroids(data,labels,k)
        if isConverged(centroids,newCentroids):
            break
        centroids = newCentroids
        
        print("第", i+1,"次迭代","用时：{0}".format(time.time() - start))
    return labels, centroids

def display_colored_clusters(image, labels, centroids):
    segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)
    plt.title("Colored Segmentation")
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

def display_separated_clusters(image, labels, k):
    h, w, c = image.shape
    fig, axes = plt.subplots(1, k, figsize=(15, 5))
    for i in range(k):
        cluster_image = np.zeros_like(image)
        cluster_image = cluster_image.reshape(-1, 3)
        cluster_image[labels == i] = image.reshape(-1, 3)[labels == i]
        cluster_image = cluster_image.reshape(h, w, c)
        
        axes[i].imshow(cluster_image)
        axes[i].set_title(f"Cluster {i + 1}")
        axes[i].axis("off")
    plt.show()

def plot_clusters_2D(data, labels):
    # 使用PCA将数据降维到2D，以便可视化
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # 绘制聚类结果
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title("2D Projection of Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

image_path = '../../data/imgs/0618.png'
k = 2
image = loadImage(image_path)
data = process_image(image)
labels, centroids = k_means(data,k)


display_colored_clusters(image, labels, centroids)

display_separated_clusters(image,labels,k)

plot_clusters_2D(data,labels)


# 计算轮廓系数
silhouette_avg = silhouette_score(data, labels)
print(f'Silhouette Score: {silhouette_avg}')

# 计算Calinski-Harabasz指数
ch_index = calinski_harabasz_score(data, labels)
print(f'Calinski-Harabasz Index: {ch_index}')

# 计算Davies-Bouldin指数
db_index = davies_bouldin_score(data, labels)
print(f'Davies-Bouldin Index: {db_index}')

# 绘制每个簇的颜色直方图
for i in range(k):
    cluster_data = data[labels == i]
    plt.figure()
    # 对于每个颜色通道绘制直方图
    for channel in range(3):  # 假设我们处理的是RGB图像
        plt.hist(cluster_data[:, channel], bins=range(256), alpha=0.5, label=f'Channel {channel+1}', histtype='stepfilled')
    plt.title(f'Color Histogram for Cluster {i + 1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()