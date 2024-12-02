# Libraries
import numpy as np                                                      # For numerical computations
from PIL import Image                                                   # For opening, manipulating, and saving images
import matplotlib.pyplot as plt                                         # For creating visualizations
from skimage.metrics import peak_signal_noise_ratio as psnr             # For measuring and evaluating image quality
import random                                                           # For generating random numbers
import argparse                                                         # For parsing command-line arguments
import time                                                             # For tracking execution time
from scipy.ndimage import median_filter                                 # For modeling errors and correcting them
from scipy.spatial.distance import cdist                                # For calculating pairwise distances between two sets of points
import os                                                               # For interacting with the operating system

#-------------------------------------------------------------------------------------------------------------
def loading(inp_path):                                                  # Function to load an image
    image = Image.open(inp_path)
    image = image.convert('RGB')                                        # Convert the image to RGB mode
    return np.array(image)

def save_func(inp_array, out_path):                                     # Function to save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image = Image.fromarray(inp_array)
    image.save(out_path)
    print(f"Image saved at {out_path}")
#-------------------------------------------------------------------------------------------------------------
def bsa_func(codebook, dist_matrix):                                    # Binary Switching Algorithm function
    n = len(codebook)                                                   # Initialize variables for BSA algorithm
    arrangment = np.arange(n)
    opt_arrangment = arrangment.copy()                                  # To store the best arrangement found
    min_distort = np.inf

    def calcu_distort_func(arrangment):                                 # Function to calculate distortion
        total_distort = 0
        for i in range(n):
            for j in range(i + 1, n):
                hamm_dist = bin(arrangment[i] ^ arrangment[j]).count('1')   # Hamming distance
                euclid_dist = dist_matrix[i, j]                             # Euclidean distance
                total_distort += hamm_dist * euclid_dist                    # Update by adding the product of two distance measures
        return total_distort

    present_distort = calcu_distort_func(arrangment)

    progress = True
    while progress:
        progress = False
        for i in range(n):                                              # Iterate over all elements in the arrangement
            for j in range(i + 1, n):                                   # Compare each element with the following elements (i+1 to n)
                arrangment[i], arrangment[j] = arrangment[j], arrangment[i] # Swap positions of elements i and j in the arrangement
                new_distort = calcu_distort_func(arrangment)            # Calculate distortion for the new arrangement
                if new_distort < present_distort:                       # Update current distortion (if the new distortion is smaller)
                    present_distort = new_distort                       # Update current distortion
                    arrangment = arrangment.copy()
                    progress = True                                     # Continue the loop
                else:
                    arrangment[i], arrangment[j] = arrangment[j], arrangment[i] # Revert elements to original positions (if no improvement)

    return opt_arrangment
#-------------------------------------------------------------------------------------------------------------
def vq_gray_func(pix, clusters, max_iter=100):                          # K-means with pseudo-Gray coding
    centroids = pix[random.sample(range(len(pix)), clusters)]

    for _ in range(max_iter):                                           # Loop for iterations
        cluster_set = assign_clusters_func(pix, centroids)              # Assign each pixel to the nearest centroid to form clusters
        new_centroids = revise_centroids_func(pix, cluster_set)         # Calculate new centroids based on the current clusters

        if np.allclose(centroids, new_centroids):                       # Check if centroids are close enough to stop
            break

        centroids = new_centroids                                       # Update centroids

    tags = np.zeros(len(pix), dtype=int)                                # Assign labels
    for i, cluster in enumerate(cluster_set):                           # Loop through each cluster
        for id_pix in cluster:                                          # Assign a cluster index to corresponding pixel IDs
            tags[id_pix] = i

    dist_matrix = cdist(centroids, centroids, metric='euclidean')       # Calculate Euclidean distances
    opt_labels = bsa_func(np.arange(len(centroids)), dist_matrix)       # Optimize cluster labels

    new_labels = np.array([opt_labels[tag] for tag in tags])            # Reassign labels accordingly

    return centroids, new_labels
#.................................................................................................................

def vq_k_func(pix, k, max_iter=100):                                    # Function for vector quantization (K-means algorithm)
    centroids = pix[random.sample(range(len(pix)), k)]
    for _ in range(max_iter):
        clusters = assign_clusters_func(pix, centroids)                 # Assign each pixel to the nearest centroid to form clusters
        present_centroids = revise_centroids_func(pix, clusters)        # Calculate new centroids
        if np.allclose(centroids, present_centroids):                   # Check if centroids are close enough
            break
        centroids = present_centroids

    Tags = np.zeros(len(pix), dtype=int)                                # Assign labels
    for i, cluster in enumerate(clusters):                              # Loop through each cluster
        for id_pix in cluster:                                          # Assign a cluster index to corresponding pixel IDs
            Tags[id_pix] = i
    print(f"K-means with k={k} is completed")
    return centroids, Tags
#..........................................................................................
def assign_clusters_func(pix, centroids):                              # Function for assigning clusters
    clusters = [[] for _ in range(len(centroids))]
    for i, pixel in enumerate(pix):
        dist = [np.linalg.norm(pixel - centroid) for centroid in centroids]  # Calculate Euclidean distance
        cluster_i = np.argmin(dist)
        clusters[cluster_i].append(i)
    return clusters
#....................................................................................................................
def revise_centroids_func(pix, clusters):                               # Update centroids for clusters
    centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            centroid = np.mean([pix[i] for i in cluster], axis=0)
        else:
            centroid = pix[random.randint(0, len(pix) - 1)]
        centroids.append(centroid)
    return np.array(centroids)
#.....................................................................................................
def compress_inp_func(pix, centroids, Tags):                            # Compress input pixels based on assigned centroids
    pix_compress = np.array([centroids[Tag] for Tag in Tags])           # Replace each pixel with its corresponding centroid
    return pix_compress
#...................................................................................................................
def metric_compress_func(original_size, clusters, image_shape):         # Calculate compression ratio
    original_size_bits = original_size * 8
    compressed_size_bits = clusters * 3 * 8
    return original_size_bits / compressed_size_bits
#....................................................................................................................
def ber_func(Tags, error_rate):                                        # Simulate a binary error rate by introducing random errors
    Tags_damaged = Tags.copy()
    num_errors = int(len(Tags) * error_rate)                           # Calculate the number of errors to introduce
    for _ in range(num_errors):                                        # Loop through the number of errors
        index = random.randint(0, len(Tags) - 1)                       # Randomly select an index in the label list
        Tags_damaged[index] = random.randint(0, np.max(Tags))          # Randomly assign a new label to the selected index
    return Tags_damaged
#....................................................................................................................
def reconst_image_func(centroids, Tags, image_shape):                  # Function for reconstructing an image
    pix_compress = np.array([centroids[Tag] for Tag in Tags])
    return pix_compress.reshape(image_shape).astype(np.uint8)
#.....................................................................................................................
def median_func(image, size=3):                                        # Apply a median filter to correct errors
    if len(image.shape) == 3:                                          # Apply the median filter (3 channels)
        return np.stack([median_filter(image[:, :, i], size=size) for i in range(3)], axis=2)  # Apply the filter to each channel (R, G, B)
    else:
        return median_filter(image, size=size)                         # Apply the median filter (single channel)
#.....................................................................................................................
def plot_image_with_title(image, title):                               # Visualization
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
#......................................................................................................................
