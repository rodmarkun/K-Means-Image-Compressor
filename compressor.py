import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def compress_image(img_path, K, max_iters):
    original_img = plt.imread(img_path)
    plt.imshow(original_img)
    if original_img.shape[2] == 4:  # Checking if the image has 4 channels
        original_img = original_img[:, :, :3]  # Keeping only the first three channels

    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

    kmeans = KMeans(n_clusters=K, max_iter=max_iters)
    kmeans.fit(X_img)
    centroids = kmeans.cluster_centers_
    idx = kmeans.labels_

    return original_img, X_img, centroids, idx

def plot_kMeans_RGB(X, centroids):
    # Plot the colors and centroids in a 3D space
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221, projection='3d')

    # First plot the color points with a lower zorder
    ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X, alpha=0.1, zorder=1)

    # Then plot the centroids with a higher zorder to ensure they appear on top
    ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3, zorder=2)

    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.yaxis.set_pane_color((0., 0., 0., .2))
    ax.set_title("Original colors and their color clusters' centroids")

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    plt.figure(figsize=(8, 2))  # Adjusted for a better aspect ratio
    plt.xticks([])
    plt.yticks([])
    plt.imshow(palette)
    plt.title("Centroid Colors")

def display_original_vs_compressed(K, original_img, centroids, idx):
    # Replace each pixel with the color of the closest centroid
    X_recovered = centroids[idx] 
    # Reshape image into proper dimensions
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Display original image
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Adjusted figsize for better side-by-side comparison
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].axis('off')  # Use axis('off') to hide axes for cleaner presentation

    # Display compressed image
    ax[1].imshow(X_recovered)
    ax[1].set_title(f'Compressed with {K} colours')
    ax[1].axis('off')

def generate_plots(X, centroids, idx, K, original_img):

    # Generate the plots
    plot_folder = 'static/plots'  # Ensure this directory exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    # Filenames for the plots
    filenames = []

    # Plot 3: Original vs Compressed Image
    plt.figure()
    display_original_vs_compressed(K, original_img, centroids, idx)
    original_vs_compressed_filename = os.path.join(plot_folder, 'original_vs_compressed.png')
    plt.savefig(original_vs_compressed_filename)
    plt.close()
    filenames.append(original_vs_compressed_filename)
    
    # Plot 1: K-Means RGB
    plt.figure()
    plot_kMeans_RGB(X, centroids)
    kmeans_rgb_filename = os.path.join(plot_folder, 'kmeans_rgb.png')
    plt.savefig(kmeans_rgb_filename)
    plt.close()
    filenames.append(kmeans_rgb_filename)

    # Plot 2: Centroid Colors
    plt.figure()
    show_centroid_colors(centroids)
    centroid_colors_filename = os.path.join(plot_folder, 'centroid_colors.png')
    plt.savefig(centroid_colors_filename)
    plt.close()
    filenames.append(centroid_colors_filename)
    
    # Return list of paths to the saved plot images
    return filenames
