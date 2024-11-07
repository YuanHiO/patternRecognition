import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans

# 读取并压缩图像
image_path = 'data/imgs/0618.png'  # 请替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("图像加载失败，请检查路径是否正确。")
else:
    # 将BGR转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. 压缩图像分辨率（例如缩小到原来25%大小）
    scale_percent = 25
    new_width = int(image_rgb.shape[1] * scale_percent / 100)
    new_height = int(image_rgb.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 获取压缩后的图像大小
    height, width, channels = resized_image.shape
    flat_image = resized_image.reshape((-1, 3))

    # 2. 使用较少的邻居数进行Isomap降维
    isomap = Isomap(n_neighbors=5, n_components=2)  # 减少邻居数量
    low_dim_data = isomap.fit_transform(flat_image)

    # 3. 分块处理 - 划分图像为若干小块，逐块降维和聚类
    n_clusters = 2  # 设置分割区域的数量
    block_size = 1000  # 定义块大小，以防止内存过载

    segmented_labels = []
    for i in range(0, low_dim_data.shape[0], block_size):
        block = low_dim_data[i:i + block_size]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(block)
        segmented_labels.extend(kmeans.labels_)

    # 将分割结果映射回图像的形状
    segmented_image = np.array(segmented_labels).reshape((height, width))

    # 显示原始图像和分割结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image)
    plt.title("Resized Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='viridis')
    plt.title("Segmented Image")
    plt.axis('off')

    plt.show()
