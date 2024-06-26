import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.cluster import KMeans
import uuid
import constants
import imageio

matplotlib.use('agg')

def compress_image(img_path, K, max_iters):
    """
    Compresses an image using K-Means clustering to reduce the number of unique colors.
    
    Parameters:
        img_path (str): Path to the image file.
        K (int): Number of clusters (colors) for K-Means clustering.
        max_iters (int): Maximum number of iterations for the K-Means algorithm.

    Returns:
        tuple: A tuple containing:
            - original_img (ndarray): Original image array.
            - X_img (ndarray): Flattened image array with shape (n, 3), where n is the number of pixels.
            - centroids (ndarray): Array of centroids, representing the dominant colors.
            - idx (ndarray): Array of indices for each pixel indicating the nearest centroid.
            - unique_filename (str): Filename where the compressed image is saved.
            - X_recovered (ndarray): Image array reconstructed from the centroids.
    """
    original_img = plt.imread(img_path)
    plt.imshow(original_img)
    if original_img.shape[2] == 4:  # Checking if the image has 4 channels
        original_img = original_img[:, :, :3]  # Keeping only the first three channels

    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

    kmeans = KMeans(n_clusters=K, max_iter=max_iters)
    kmeans.fit(X_img)
    centroids = kmeans.cluster_centers_
    idx = kmeans.labels_

    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Save to PNG
    X_recovered_uint8 = np.clip(X_recovered * 255, 0, 255).astype(np.uint8)
    unique_filename = f"compressed_image_{uuid.uuid4()}.png"
    image_path = os.path.join(constants.COMPRESSED_FOLDER, unique_filename)
    imageio.imwrite(image_path, X_recovered_uint8) 

    return original_img, X_img, centroids, idx, unique_filename, X_recovered

def plot_kMeans_RGB(X, centroids):
    """
    Creates a 3D scatter plot of the original colors and their corresponding K-Means centroids.
    
    Parameters:
        X (ndarray): Flattened image data used for K-Means clustering.
        centroids (ndarray): Array of centroids computed by K-Means clustering.

    Returns:
        matplotlib.figure.Figure: A figure object containing the plot.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X, alpha=0.1, zorder=1)
    ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3, zorder=2)

    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.yaxis.set_pane_color((0., 0., 0., .2))
    ax.set_title("Original colors and their color clusters' centroids")
    fig.tight_layout()
    return fig

def show_centroid_colors(centroids):
    """
    Displays a palette of centroid colors sorted by luminance.

    Parameters:
        centroids (ndarray): Array of centroids, representing the dominant colors.
    
    Returns:
        matplotlib.figure.Figure: A figure object containing the color palette.
    """
    lum = 0.299 * centroids[:, 0] + 0.587 * centroids[:, 1] + 0.114 * centroids[:, 2]
    sorted_centroids = centroids[np.argsort(lum)]


    fig, ax = plt.subplots(figsize=(8, 2))
    palette = np.expand_dims(sorted_centroids, axis=0)
    ax.imshow(palette)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Centroid Colors")
    fig.tight_layout()
    return fig

def display_original_vs_compressed(K, original_img, centroids, idx):
    """
    Displays side-by-side comparison of the original and compressed images.

    Parameters:
        K (int): Number of color clusters used in compression.
        original_img (ndarray): The original image array.
        centroids (ndarray): Array of centroids representing the compressed colors.
        idx (ndarray): Array of indices for each pixel indicating the nearest centroid.

    Returns:
        matplotlib.figure.Figure: A figure object containing the side-by-side comparison.
    """
    X_recovered = centroids[idx] 
    X_recovered = np.reshape(X_recovered, original_img.shape)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(original_img)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(X_recovered)
    axs[1].set_title(f'Compressed with {K} colours')
    axs[1].axis('off')
    fig.tight_layout()
    return fig

def plot_quantization_error(original_img, X_recovered):
    """
    Calculates and displays the quantization error between the original and compressed images.
    
    Parameters:
        original_img (ndarray): The original uncompressed image.
        X_recovered (ndarray): The compressed image reconstructed from centroids.

    Returns:
        matplotlib.figure.Figure: A figure object displaying the quantization error.
    """
    error = np.abs(original_img - X_recovered)
    error_magnitude = np.sqrt(np.sum(error**2, axis=2))
    error_normalized = error_magnitude / np.max(error_magnitude)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(error_normalized, cmap='gray')
    ax.set_title('Quantization Error')
    ax.axis('off')
    fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    return fig

def generate_plots(X, centroids, idx, K, original_img, folder_name, X_recovered):
    """
    Generates and saves multiple visualizations related to the image compression process.

    Parameters:
        X (ndarray): Flattened image data used for K-Means clustering.
        centroids (ndarray): Array of centroids computed by K-Means clustering.
        idx (ndarray): Array of indices for each pixel indicating the nearest centroid.
        K (int): Number of color clusters used in compression.
        original_img (ndarray): The original uncompressed image.
        folder_name (str): Folder name where the plots will be saved.
        X_recovered (ndarray): The compressed image reconstructed from centroids.

    Returns:
        list of str: A list containing the paths to the saved plot files.
    """
    plot_folder = f'static/plots/{folder_name}'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    filenames = []

    # Plot 1: Original vs Compressed Image
    fig1 = display_original_vs_compressed(K, original_img, centroids, idx)
    original_vs_compressed_filename = os.path.join(plot_folder, 'original_vs_compressed.png')
    fig1.savefig(original_vs_compressed_filename, bbox_inches='tight', transparent=True)
    fig1.clf()
    filenames.append(original_vs_compressed_filename)

    # Plot 2: Centroid Colors
    fig2 = show_centroid_colors(centroids)
    centroid_colors_filename = os.path.join(plot_folder, 'centroid_colors.png')
    fig2.savefig(centroid_colors_filename, bbox_inches='tight', transparent=True)
    fig2.clf()
    filenames.append(centroid_colors_filename)

    # Plot 3: K-Means RGB
    fig3 = plot_kMeans_RGB(X, centroids)
    kmeans_rgb_filename = os.path.join(plot_folder, 'kmeans_rgb.png')
    fig3.savefig(kmeans_rgb_filename, bbox_inches='tight', transparent=True)
    fig3.clf()
    filenames.append(kmeans_rgb_filename)

    # Plot 4: Quantization Error
    fig_error = plot_quantization_error(original_img, X_recovered)
    quantization_error_filename = os.path.join(plot_folder, 'quantization_error.png')
    fig_error.savefig(quantization_error_filename, bbox_inches='tight', transparent=True)
    fig_error.clf()
    filenames.append(quantization_error_filename)
    
    plt.close('all')  # Close all figures to free memory
    return filenames
