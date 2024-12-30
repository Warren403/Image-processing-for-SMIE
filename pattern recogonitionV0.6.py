import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tkinter import filedialog
from tkinter import Tk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
import time

# Fixing random seed
# Set a fixed random seed for reproducibility
np.random.seed(42)

def extract_features(spectra, wavelengths):
    # This function computes various features from each spectrum:
    # - Kurtosis, Skewness
    # - Peak and Valley values (including their counts and distances)
    # - Global max/min values
    # - Local curvature near maxima/minima
    # - Slope between global max and min
    curvature_features = []

    for spectrum in spectra:
        if len(spectrum) < 3:
            # If the length of the spectrum is less than 3, fill with zeros to keep dimension consistent
            curvature_features.append(np.zeros(13))
            continue
        
        # Calculate first and second derivatives
        first_derivative = np.gradient(spectrum)
        second_derivative = np.gradient(first_derivative)
        
        # Find global max and min positions
        max_pos = np.argmax(spectrum)
        min_pos = np.argmin(spectrum)
        
        # Corresponding global max and min values
        max_value = spectrum[max_pos]
        min_value = spectrum[min_pos]
        
        # Calculate local maximum curvature around the global maxima
        if max_pos - 1 >= 0 and max_pos + 2 <= len(second_derivative):
            max_curvature = np.max(np.abs(second_derivative[max_pos-1:max_pos+2]))
        else:
            max_curvature = 0
        
        # Calculate local maximum curvature around the global minima
        if min_pos - 1 >= 0 and min_pos + 2 <= len(second_derivative):
            min_curvature = np.max(np.abs(second_derivative[min_pos-1:min_pos+2]))
        else:
            min_curvature = 0
        
        # Slope between global max and global min
        if max_pos != min_pos:
            slope = (max_value - min_value) / (wavelengths[max_pos] - wavelengths[min_pos])
        else:
            slope = 0
        
        # Calculate kurtosis and skewness of the spectrum
        kurtosis_value = kurtosis(spectrum)
        skewness_value = skew(spectrum)

        # Find peaks and valleys (using negative spectrum for valley detection)
        peaks, _ = find_peaks(spectrum)
        valleys, _ = find_peaks(-spectrum)
        num_peaks = len(peaks)
        num_valleys = len(valleys)
        
        # Peak-to-peak distance and peak value
        if num_peaks > 0:
            peak_to_peak_dist = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
            peak_value = np.max(spectrum[peaks])
        else:
            peak_to_peak_dist = 0
            peak_value = 0
        
        # Valley-to-valley distance and valley value
        if num_valleys > 0:
            valley_to_valley_dist = np.mean(np.diff(valleys)) if num_valleys > 1 else 0
            valley_value = np.min(spectrum[valleys])
        else:
            valley_to_valley_dist = 0
            valley_value = 0

        # Compile all extracted features into an array
        features = np.array([
            kurtosis_value, skewness_value, peak_value, valley_value,
            num_peaks, num_valleys, peak_to_peak_dist, valley_to_valley_dist,
            max_value, min_value, max_curvature, min_curvature, slope
        ])
        curvature_features.append(features)
    
    curvature_features = np.array(curvature_features)
    
    return curvature_features


def display_image_comparison(original, blurred, title, position):
    # This function displays two images side by side for visual comparison:
    # the original (rotated) image and the Gaussian blurred one.
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))  # Increased size for better visualization
    fig.suptitle(title)
    im1 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Image')
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.75)  # Shorten the colorbar
    
    im2 = axes[1].imshow(blurred, cmap='viridis')
    axes[1].set_title('Gaussian Blurred Image')
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.75)  # Shorten the colorbar

    # Position the figure window on the screen
    fig.canvas.manager.window.wm_geometry("+%d+%d" % (position[0], position[1]))
    plt.show(block=False)
    plt.pause(1)
    plt.close(fig)

def read_and_display_images(directory, prefix, sigma, display_images):
    # This function reads all TXT files in the specified directory matching the pattern (prefix_*.txt).
    # Each file is read as a 2D array, rotated, and then Gaussian blurred.
    # The blurred images are stored in a dictionary, and their pixel-wise spectra are stored in another dictionary.
    print(f"Searching for files in {directory} with prefix {prefix}")
    path_pattern = os.path.join(directory, f"{prefix}_*.txt")
    files = sorted(glob.glob(path_pattern), key=lambda x: int(os.path.basename(x).replace('.txt', '').split('_')[1].replace('nm', '')))
    
    if not files:
        print(f"No files found with pattern: {path_pattern}")
        return {}, {}

    data = {}
    spectrum_data = {}

    position = (100, 100)

    for file_path in files:
        print(f"Processing file: {file_path}")

        with open(file_path, 'r') as file:
            filename = os.path.basename(file_path)
            wavelength = filename.replace(prefix + '_', '').replace('.txt', '')
            content = file.readlines()
            if not content:
                print(f"No data in file {filename}")
                continue

            # Convert each line in the file to floats and form a 2D array
            image_array = np.array([list(map(float, line.strip().split())) for line in content if line.strip()])
            # Rotate the array 90 degrees
            rotated_image = np.rot90(image_array, k=1)
            # Apply Gaussian blur
            blurred_image = gaussian_filter(rotated_image, sigma=sigma)
            data[wavelength] = blurred_image

            # Initialize the spectrum_data structure if empty
            if not spectrum_data:
                for i in range(blurred_image.shape[0]):
                    for j in range(blurred_image.shape[1]):
                        spectrum_data[(i, j)] = []
            # For each pixel, append (wavelength, intensity) to build full spectra
            for i in range(blurred_image.shape[0]):
                for j in range(blurred_image.shape[1]):
                    spectrum_data[(i, j)].append((wavelength, blurred_image[i, j]))

            # Display original and blurred images if user chooses
            if display_images:
                display_image_comparison(rotated_image, blurred_image, f"Wavelength: {wavelength}", position)

    return data, spectrum_data

def plot_all_spectra(spectrum_data):
    # This function collects spectra from all pixels and returns them as a 2D array (rows = pixels, columns = wavelengths),
    # along with the sorted list of wavelengths.
    if not spectrum_data:
        print("No spectrum data to plot.")
        return [], []

    all_spectra = []
    for position, spectrum in spectrum_data.items():
        wavelengths, intensities = zip(*spectrum)
        # Convert the wavelength strings (e.g., "123nm") to float
        wavelengths = [float(w.replace('nm', '')) for w in wavelengths]
        all_spectra.append(intensities)
    
    return np.array(all_spectra), wavelengths

def plot_spectra_comparison(original_spectra, smoothed_spectra, wavelengths):
    # This function plots the original vs. smoothed spectra side by side for each pixel.
    plt.figure(figsize=(10, 5))
    for i in range(len(original_spectra)):
        plt.subplot(1, 2, 1)
        plt.plot(wavelengths, original_spectra[i], alpha=0.5)
        plt.title('Original Spectra')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Delta')
        
        plt.subplot(1, 2, 2)
        plt.plot(wavelengths, smoothed_spectra[i], alpha=0.5)
        plt.title('Smoothed Spectra')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Delta')
        
        # Mark the global maximum of the smoothed spectrum
        max_pos = np.argmax(smoothed_spectra[i])
        plt.plot(wavelengths[max_pos], smoothed_spectra[i][max_pos], 'ko', markerfacecolor='none', markersize=5)
    
    plt.tight_layout()
    plt.show()

def format_with_ellipsis(matrix, num_elements=5):
    # Utility function to limit the display of matrix rows and columns,
    # replacing the middle with "..." for brevity.
    formatted_matrix = []
    for row in matrix:
        formatted_row = [f"{value:.2f}" for value in row]
        if len(row) > 2 * num_elements:
            formatted_row = formatted_row[:num_elements] + ["..."] + formatted_row[-num_elements:]
        formatted_matrix.append(" ".join(formatted_row))
    return formatted_matrix

def print_with_ellipsis(matrix, num_elements=5):
    # Print the first few and last few rows of a matrix with ellipsis in between.
    formatted_matrix = format_with_ellipsis(matrix, num_elements)
    total_rows = len(formatted_matrix)
    
    # Print the first 5 rows
    for row in formatted_matrix[:5]:
        print(row)
    
    # Print ellipsis if there are more than 10 rows
    if total_rows > 10:
        print("...")
    
    # Print the last 5 rows
    for row in formatted_matrix[-5:]:
        print(row)

def cluster_spectra(spectra, wavelengths, n_clusters):
    # This function performs the following steps for clustering:
    # 1. Smooth the spectra using Savitzky-Golay filter.
    # 2. Extract multiple features per spectrum (e.g., curvature, peaks, valleys).
    # 3. Normalize features using StandardScaler.
    # 4. Reduce dimensionality to 3D using PCA.
    # 5. Cluster the reduced features using KMeans.
    if not spectra.size:
        print("No spectra data to cluster.")
        return np.array([]), np.array([]), np.array([])  # Return three empty arrays

    # Apply Savitzky-Golay filter for smoothing
    smoothed_spectra = savgol_filter(spectra, window_length=11, polyorder=3, axis=1)

    # Extract features
    features = extract_features(smoothed_spectra, np.array(wavelengths))

    # Standardize the features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    print("\nNormalized Features:")
    print_with_ellipsis(features_normalized)

    # Perform PCA to reduce features to 3 principal components
    pca = PCA(n_components=3)
    features_reduced = pca.fit_transform(features_normalized)

    print("\nPCA Reduced Features:")
    print_with_ellipsis(features_reduced, num_elements=3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42)
    labels = kmeans.fit_predict(features_reduced)

    return labels, features_reduced

def remap_labels(labels):
    # This function remaps cluster labels so that the most frequent cluster is labeled 0, next frequent is 1, and so on.
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_labels = [label for label, _ in sorted(zip(unique_labels, counts), key=lambda x: x[1], reverse=True)]
    label_mapping = {old: new for new, old in enumerate(sorted_labels)}
    remapped_labels = np.vectorize(label_mapping.get)(labels)
    return remapped_labels, label_mapping

def plot_classified_image_and_pca(data, labels, features_reduced, shape, n_clusters):
    # This function visualizes the clustering results in two ways:
    # 1. Displays an image where each pixel is colored by its cluster label.
    # 2. Plots a 3D PCA scatter plot color-coded by cluster.
    if len(labels) == 0:
        print("No classification labels to plot.")
        return None

    remapped_labels, label_mapping = remap_labels(labels)
    
    unique_labels = np.unique(remapped_labels)
    num_labels = len(unique_labels)

    colors = plt.cm.get_cmap('viridis', n_clusters)
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Plot the classified image
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    classified_image = np.zeros(shape)
    index = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if index >= len(remapped_labels):
                break
            label = remapped_labels[index]
            classified_image[i, j] = label if label != -1 else -1
            index += 1

    im = ax1.imshow(classified_image, cmap=colors, origin='upper', aspect='auto', vmin=0, vmax=n_clusters - 1)
    fig1.colorbar(im, ax=ax1, ticks=np.arange(n_clusters))
    ax1.set_title('Classified Image')
    
    # Position the figure window on the screen
    fig1.canvas.manager.window.wm_geometry("+100+100")

    # Save classified image to a TXT file
    classified_filename = f"classified_K={n_clusters}.txt"
    np.savetxt(classified_filename, classified_image.astype(int), fmt="%d")  # Using integer format
    print(f"Classified image saved as {classified_filename}")

    # Plot 3D PCA
    fig2 = plt.figure(figsize=(7, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    for label in unique_labels:
        indices = np.where(remapped_labels == label)
        ax2.scatter(features_reduced[indices, 0], features_reduced[indices, 1], features_reduced[indices, 2], 
                    label=f'Cluster {label}', color=color_map[label], s=10)

    ax2.set_xlabel('Principle Component 1')
    ax2.set_ylabel('Principle Component 2')
    ax2.set_zlabel('Principle Component 3')
    ax2.set_title('3D PCA Space', pad=20)

    # Place legend outside the plot
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    # Position the figure window on the screen
    fig2.canvas.manager.window.wm_geometry("+800+100")

    plt.show()

    return remapped_labels, color_map

def plot_average_spectra(spectra, labels, wavelengths, color_map):
    # This function calculates and plots the average spectrum of each cluster.
    unique_labels = np.unique(labels)
    avg_spectra = {}

    # Compute the average spectrum for each cluster
    for label in unique_labels:
        label_indices = np.where(labels == label)
        avg_spectra[label] = np.mean(spectra[label_indices], axis=0)

    linestyles = ['-', '--', '-.', ':']
    plt.figure(figsize=(10, 6))
    for i, (label, avg_spectrum) in enumerate(avg_spectra.items()):
        plt.plot(wavelengths, avg_spectrum, label=f'Cluster {label}', color=color_map[label], linestyle=linestyles[label % len(linestyles)])
    
    plt.title('Average Spectra for Each Cluster')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Delta')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# Main execution flow
root = Tk()
root.withdraw()

selected_directory = filedialog.askdirectory(title="Select image data folder")
if selected_directory:
    print(f"Selected directory: {selected_directory}")
    folder_name = os.path.basename(selected_directory)
    if '_' in folder_name:
        prefix = folder_name.split('_')[1]
    else: 
        print("Folder name does not contain an underscore. Please select a valid folder.")
        exit()

    print(f"Processing folder: {folder_name} with prefix: {prefix}")
    
    # Ask user if they want to display original & Gaussian blurred image comparisons
    display_images_choice = input("Do you want to display original and Gaussian blurred image comparison? (Y/N): ").strip().upper()
    display_images = (display_images_choice == 'Y')

    # Read and display images
    data, spectrum_data = read_and_display_images(selected_directory, prefix, sigma=1.2, display_images=display_images)

    if spectrum_data:
        print("Plotting all spectra...")
        all_spectra, wavelengths = plot_all_spectra(spectrum_data)

        # Ask user if they want to compare original vs. smoothed spectra for all pixels
        spectra_comparison_choice = input("Do you want to plot spectra comparison for all pixels? (Y/N): ").strip().upper()
        if spectra_comparison_choice == 'Y':
            plot_spectra_comparison(all_spectra, savgol_filter(all_spectra, window_length=11, polyorder=3, axis=1), wavelengths)

        # Get cluster range from user
        k_min = int(input("Enter the minimum number of clusters (K): "))
        k_max = int(input("Enter the maximum number of clusters (K): "))
        
        print("Clustering spectra...")
        for n_clusters in range(k_min, k_max + 1):
            print(f"Clustering with {n_clusters} clusters...")
            labels, features_reduced = cluster_spectra(all_spectra, wavelengths, n_clusters)

            if labels.size > 0 and features_reduced.size > 0:
                remapped_labels, color_map = plot_classified_image_and_pca(data, labels, features_reduced, list(data.values())[0].shape, n_clusters)
                plot_average_spectra(all_spectra, remapped_labels, wavelengths, color_map)
            else:
                print(f"No valid clustering results for {n_clusters} clusters.")
else:
    print("No directory selected. Exiting...")

root.destroy()
