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

    # Convertimos X_recovered a uint8 para su correcto guardado como imagen PNG
    X_recovered_uint8 = np.clip(X_recovered * 255, 0, 255).astype(np.uint8)
    
    # Generamos un nombre de archivo único y guardamos la imagen
    unique_filename = f"compressed_image_{uuid.uuid4()}.png"
    image_path = os.path.join(constants.COMPRESSED_FOLDER, unique_filename)
    imageio.imwrite(image_path, X_recovered_uint8)  # Ahora correctamente guardamos X_recovered como imagen

    return original_img, X_img, centroids, idx, unique_filename

def plot_kMeans_RGB(X, centroids):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X, alpha=0.1, zorder=1)
    ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3, zorder=2)

    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.yaxis.set_pane_color((0., 0., 0., .2))
    ax.set_title("Original colors and their color clusters' centroids")
    plt.tight_layout()

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    plt.figure(figsize=(8, 2))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(palette)
    plt.title("Centroid Colors")
    plt.tight_layout()

def display_original_vs_compressed(K, original_img, centroids, idx):
    X_recovered = centroids[idx] 
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Display original image
    fig, ax = plt.subplots(1, 2, figsize=(16, 8)) 
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].axis('off') 

    # Display compressed image
    ax[1].imshow(X_recovered)
    ax[1].set_title(f'Compressed with {K} colours')
    ax[1].axis('off')
    plt.tight_layout()

def generate_plots(X, centroids, idx, K, original_img, folder_name):
    plot_folder = f'static/plots/{folder_name}' 
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    filenames = []

    # Plot 1: Original vs Compressed Image
    plt.figure()
    display_original_vs_compressed(K, original_img, centroids, idx)
    original_vs_compressed_filename = os.path.join(plot_folder, 'original_vs_compressed.png')
    plt.savefig(original_vs_compressed_filename, bbox_inches='tight', transparent=True)
    plt.close()
    filenames.append(original_vs_compressed_filename)

    # Plot 3: Centroid Colors
    plt.figure()
    show_centroid_colors(centroids)
    centroid_colors_filename = os.path.join(plot_folder, 'centroid_colors.png')
    plt.savefig(centroid_colors_filename, bbox_inches='tight', transparent=True)
    plt.close()
    filenames.append(centroid_colors_filename)

    # Plot 2: K-Means RGB
    plt.figure()
    plot_kMeans_RGB(X, centroids)
    kmeans_rgb_filename = os.path.join(plot_folder, 'kmeans_rgb.png')
    plt.savefig(kmeans_rgb_filename, bbox_inches='tight', transparent=True)
    plt.close()
    filenames.append(kmeans_rgb_filename)
    
    return filenames


