import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# Define parameters for foreground (road) and background (non-road) based on prior knowledge or statistics
foreground_params = {
    'mean_rgb': np.array([150, 150, 150]),
    'std_rgb': np.array([20, 20, 20]),
}

background_params = {
    'mean_rgb': np.array([60, 60, 60]),
    'std_rgb': np.array([10, 10, 10]),
}

# Probability calculation function for each pixel using log probabilities
def calculate_log_probability(pixel, mean, std):
    log_prob = -0.5 * ((pixel - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
    return np.sum(log_prob)  # Return the sum of log probabilities across RGB channels

# Bayesian segmentation function with incremental evidence fusion using log probabilities
def bayesian_segment_incremental(image, fg_params, bg_params, block_size=3):
    height, width, _ = image.shape
    segmented_image = np.zeros((height, width), dtype=np.uint8)

    # Loop over each block in the image
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Initialize log posteriors to 0 (log(1))
            log_posterior_fg = 0
            log_posterior_bg = 0

            # Iterate through each pixel in the block
            for m in range(block_size):
                for n in range(block_size):
                    # Get pixel within the block
                    pixel = image[i + m, j + n]

                    # Calculate log likelihoods for foreground and background
                    log_p_fg = calculate_log_probability(pixel, fg_params['mean_rgb'], fg_params['std_rgb'])
                    log_p_bg = calculate_log_probability(pixel, bg_params['mean_rgb'], bg_params['std_rgb'])

                    # Update log posteriors based on pixel evidence
                    log_posterior_fg += log_p_fg
                    log_posterior_bg += log_p_bg

            # Final decision for the block based on accumulated evidence
            label = 1 if log_posterior_fg > log_posterior_bg else 0

            # Assign label to the entire block for smoother segmentation output
            segmented_image[i:i + block_size, j:j + block_size] = label * 255

    return segmented_image


# Load and prepare the image
image_rgb = cv2.cvtColor(cv2.imread('../../data/imgs/0618.png'), cv2.COLOR_BGR2RGB)

# Apply Bayesian segmentation with incremental fusion
segmented_image = bayesian_segment_incremental(image_rgb, foreground_params, background_params)

# Display segmented image
plt.imshow(segmented_image, cmap='gray')
plt.title("Bayesian Segmented Image with Incremental Evidence Fusion (Log Probabilities)")
plt.show()

# Post-process the segmented image to reduce noise
def post_process(segmented_image, open_kernel_size=3, close_kernel_size=5, min_area=200):
    # Morphological operations to clean up segmentation
    # 打开操作，去除小块噪声
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    opened = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, open_kernel)

    # 闭运算，填补小孔和细小空隙
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # opened = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    processed_image = np.zeros_like(segmented_image, dtype=np.uint8)
    # Retain large connected components
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            processed_image[labels == i] = 255
    return processed_image
# Apply post-processing
processed_segmented_image = post_process(segmented_image)

mask = cv2.imread('../../data/imgs/mask0618.png', cv2.IMREAD_GRAYSCALE)
# Flatten the ground truth and predicted images for evaluation
ground_truth = mask.flatten() // 255  # Assuming mask is in 0 and 255, convert to 0 and 1
predicted = processed_segmented_image.flatten() // 255  # Assuming segmented result is also in 0 and 255

# Calculate accuracy, precision, recall, and confusion matrix
accuracy = accuracy_score(ground_truth, predicted)
precision = precision_score(ground_truth, predicted)
recall = recall_score(ground_truth, predicted)
conf_matrix = confusion_matrix(ground_truth, predicted)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)

# Display metrics
plt.imshow(processed_segmented_image, cmap='gray')
plt.title(f"Processed Segmented Image\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
plt.show()