import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
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

# Save the original image shape
original_shape = image.shape

# Initialize the Isomap object for dimensionality reduction
isomap = Isomap(n_neighbors=5, n_components=2)

# RGB
# Resize the RGB image for faster processing
image_resized_RGB = resize(image_RGB, (image_RGB.shape[0] // 4, image_RGB.shape[1] // 4), anti_aliasing=True)
# Flatten the image for Isomap processing
image_flat_isomap_RGB = image_resized_RGB.reshape(-1, 3)
# Apply Isomap dimensionality reduction
image_reduced_isomap_RGB = isomap.fit_transform(image_flat_isomap_RGB)

# LAB
# Resize the LAB image for faster processing
image_resized_LAB = resize(image_LAB, (image_LAB.shape[0] // 4, image_LAB.shape[1] // 4), anti_aliasing=True)
# Flatten the image for Isomap processing
image_flat_isomap_LAB = image_resized_LAB.reshape(-1, 3)
# Apply Isomap dimensionality reduction
image_reduced_isomap_LAB = isomap.fit_transform(image_flat_isomap_LAB)

# HSV
# Resize the HSV image for faster processing
image_resized_HSV = resize(image_HSV, (image_HSV.shape[0] // 4, image_HSV.shape[1] // 4), anti_aliasing=True)
# Flatten the image for Isomap processing
image_flat_isomap_HSV = image_resized_HSV.reshape(-1, 3)
# Apply Isomap dimensionality reduction
image_reduced_isomap_HSV = isomap.fit_transform(image_flat_isomap_HSV)

# k-means cluster
n_clusters = 2 

# RGB
# Perform k-means clustering on the RGB image
kmeans_RGB = KMeans(n_clusters=n_clusters, random_state=42)
labels_RGB = kmeans_RGB.fit_predict(image_reduced_isomap_RGB)
# Reshape the labels to the image shape
segmented_image_RGB = labels_RGB.reshape(image_resized_RGB.shape[0], image_resized_RGB.shape[1])

# LAB
# Perform k-means clustering on the LAB image
kmeans_LAB = KMeans(n_clusters=n_clusters, random_state=42)
labels_LAB = kmeans_LAB.fit_predict(image_reduced_isomap_LAB)
# Reshape the labels to the image shape
segmented_image_LAB = labels_LAB.reshape(image_resized_LAB.shape[0], image_resized_LAB.shape[1])

# HSV
# Perform k-means clustering on the HSV image
kmeans_HSV = KMeans(n_clusters=n_clusters, random_state=42)
labels_HSV = kmeans_HSV.fit_predict(image_reduced_isomap_HSV)
# Reshape the labels to the image shape
segmented_image_HSV = labels_HSV.reshape(image_resized_HSV.shape[0], image_resized_HSV.shape[1])

# Calculate evaluation metrics for RGB
silhouette_avg_RGB = silhouette_score(image_reduced_isomap_RGB, labels_RGB)
calinski_harabasz_RGB = calinski_harabasz_score(image_reduced_isomap_RGB, labels_RGB)
davies_bouldin_RGB = davies_bouldin_score(image_reduced_isomap_RGB, labels_RGB)

# Calculate evaluation metrics for LAB
silhouette_avg_LAB = silhouette_score(image_reduced_isomap_LAB, labels_LAB)
calinski_harabasz_LAB = calinski_harabasz_score(image_reduced_isomap_LAB, labels_LAB)
davies_bouldin_LAB = davies_bouldin_score(image_reduced_isomap_LAB, labels_LAB)

# Calculate evaluation metrics for HSV
silhouette_avg_HSV = silhouette_score(image_reduced_isomap_HSV, labels_HSV)
calinski_harabasz_HSV = calinski_harabasz_score(image_reduced_isomap_HSV, labels_HSV)
davies_bouldin_HSV = davies_bouldin_score(image_reduced_isomap_HSV, labels_HSV)

# Print the evaluation metrics
print(f"RGB Evaluation Metrics:")
print(f"Silhouette Coefficient: {silhouette_avg_RGB}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_RGB}")
print(f"Davies-Bouldin Index: {davies_bouldin_RGB}")

print(f"\nLAB Evaluation Metrics:")
print(f"Silhouette Coefficient: {silhouette_avg_LAB}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_LAB}")
print(f"Davies-Bouldin Index: {davies_bouldin_LAB}")

print(f"\nHSV Evaluation Metrics:")
print(f"Silhouette Coefficient: {silhouette_avg_HSV}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_HSV}")
print(f"Davies-Bouldin Index: {davies_bouldin_HSV}")

# Display the original and segmented images
plt.subplot(3, 2, 1)
plt.imshow(image)
plt.title("Original Image RGB")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(segmented_image_RGB, cmap='viridis')
plt.title("Segmented ImageRGB isomap")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(image_LAB)
plt.title("Original Image LAB")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(segmented_image_LAB, cmap='viridis')
plt.title("Segmented ImageLAB isomap")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.imshow(image_HSV)
plt.title("Original Image HSV")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(segmented_image_HSV, cmap='viridis')
plt.title("Segmented ImageHSV isomap")
plt.axis("off")

plt.show()