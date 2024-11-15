import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# Load the image path
image_path = '../../data/imgs/0618.png'
image = cv2.imread(image_path)

# Check if the image is successfully loaded
if image is None:
    print("image read error")
else:
    # Convert the image to RGB, LAB, and HSV color spaces
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Initialize the Locally Linear Embedding (LLE) model
lle = LocallyLinearEmbedding(n_neighbors=75, n_components=2, method='standard')

# Get the original image shape
original_shape = image.shape

# RGB
# Resize the image to reduce computational complexity
image_resized_RGB = resize(image_RGB, (image_RGB.shape[0] // 2, image_RGB.shape[1] // 2), anti_aliasing=True)
# Flatten the image for LLE processing
image_flat_lle_RGB = image_resized_RGB.reshape(-1, 3)
# Apply LLE dimensionality reduction
image_reduced_lle_RGB = lle.fit_transform(image_flat_lle_RGB)

# LAB
# Resize the LAB image
image_resized_LAB = resize(image_LAB, (image_LAB.shape[0] // 2, image_LAB.shape[1] // 2), anti_aliasing=True)
# Flatten the LAB image for LLE processing
image_flat_lle_LAB = image_resized_LAB.reshape(-1, 3)
# Apply LLE dimensionality reduction
image_reduced_lle_LAB = lle.fit_transform(image_flat_lle_LAB)

# HSV
# Resize the HSV image
image_resized_HSV = resize(image_HSV, (image_HSV.shape[0] // 2, image_HSV.shape[1] // 2), anti_aliasing=True)
# Flatten the HSV image for LLE processing
image_flat_lle_HSV = image_resized_HSV.reshape(-1, 3)
# Apply LLE dimensionality reduction
image_reduced_lle_HSV = lle.fit_transform(image_flat_lle_HSV)

# k-means cluster
n_clusters = 2  

# RGB
# Perform k-means clustering on the RGB image
kmeans_RGB = KMeans(n_clusters=n_clusters, random_state=42)
labels_RGB = kmeans_RGB.fit_predict(image_reduced_lle_RGB)
# Reshape the labels to the image shape
segmented_image_RGB = labels_RGB.reshape(image_resized_RGB.shape[0], image_resized_RGB.shape[1])

# LAB
# Perform k-means clustering on the LAB image
kmeans_LAB = KMeans(n_clusters=n_clusters, random_state=42)
labels_LAB = kmeans_LAB.fit_predict(image_reduced_lle_LAB)
# Reshape the labels to the image shape
segmented_image_LAB = labels_LAB.reshape(image_resized_LAB.shape[0], image_resized_LAB.shape[1])

# HSV
# Perform k-means clustering on the HSV image
kmeans_HSV = KMeans(n_clusters=n_clusters, random_state=42)
labels_HSV = kmeans_HSV.fit_predict(image_reduced_lle_HSV)
# Reshape the labels to the image shape
segmented_image_HSV = labels_HSV.reshape(image_resized_HSV.shape[0], image_resized_HSV.shape[1])

# Calculate evaluation metrics
def calculate_metrics(data, labels):
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    return silhouette, calinski_harabasz, davies_bouldin

# RGB
silhouette_RGB, calinski_harabasz_RGB, davies_bouldin_RGB = calculate_metrics(image_reduced_lle_RGB, labels_RGB)
print(f"RGB - Silhouette Score: {silhouette_RGB}, Calinski-Harabasz Index: {calinski_harabasz_RGB}, Davies-Bouldin Index: {davies_bouldin_RGB}")

# LAB
silhouette_LAB, calinski_harabasz_LAB, davies_bouldin_LAB = calculate_metrics(image_reduced_lle_LAB, labels_LAB)
print(f"LAB - Silhouette Score: {silhouette_LAB}, Calinski-Harabasz Index: {calinski_harabasz_LAB}, Davies-Bouldin Index: {davies_bouldin_LAB}")

# HSV
silhouette_HSV, calinski_harabasz_HSV, davies_bouldin_HSV = calculate_metrics(image_reduced_lle_HSV, labels_HSV)
print(f"HSV - Silhouette Score: {silhouette_HSV}, Calinski-Harabasz Index: {calinski_harabasz_HSV}, Davies-Bouldin Index: {davies_bouldin_HSV}")

# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(3, 2, 1)
plt.imshow(image)
plt.title("Original Image RGB")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(segmented_image_RGB, cmap='viridis')
plt.title("Segmented ImageRGB LLE")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(image_LAB)
plt.title("Original Image LAB")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(segmented_image_LAB, cmap='viridis')
plt.title("Segmented ImageLAB LLE")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.imshow(image_HSV)
plt.title("Original Image HSV")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(segmented_image_HSV, cmap='viridis')
plt.title("Segmented ImageHSV LLE")
plt.axis("off")

plt.show()