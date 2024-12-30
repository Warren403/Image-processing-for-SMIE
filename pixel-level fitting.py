import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tkinter import filedialog
from tkinter import Tk
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator


# File paths
si_file_path = r'E:\image processing for elliposmetry\Si.txt'
sio2_file_path = r'E:\image processing for elliposmetry\SiO2.txt'

# Load Si and SiO2 optical constants
print(f"Loading Si optical constants from: {si_file_path}")
wavelength_si, n_si, k_si = np.loadtxt(si_file_path, unpack=True)

print(f"Loading SiO2 optical constants from: {sio2_file_path}")
wavelength_sio2, n_sio2, k_sio2 = np.loadtxt(sio2_file_path, unpack=True)

# Air refractive index 
N_air = 1.0 + 0j
# Incident angle of 40 degrees
theta = np.radians(40)  # Incident angle of 40 degrees

# Snell's law for complex refractive indices
def snell_law_complex(N1, theta1, N2):
    return np.arcsin((N1 * np.sin(theta1)) / N2)

# Fresnel coefficients for p-polarized and s-polarized light
def fresnel_coefficients(N1, N2, theta1, theta2):
    rp = (N2 * np.cos(theta1) - N1 * np.cos(theta2)) / (N2 * np.cos(theta1) + N1 * np.cos(theta2))
    rs = (N1 * np.cos(theta1) - N2 * np.cos(theta2)) / (N1 * np.cos(theta1) + N2 * np.cos(theta2))
    return rp, rs

# Phase shift factor beta
def compute_beta(d, N, theta, wavelength):
    return (2 * np.pi * d * N * np.cos(theta)) / wavelength

# Single-layer reflectance calculation considering multiple reflections
def single_layer_reflectance(N0, N1, N2, d, wavelength, theta):
    theta1 = snell_law_complex(N0, theta, N1)
    theta2 = snell_law_complex(N1, theta1, N2)
    
    r01_p, r01_s = fresnel_coefficients(N0, N1, theta, theta1)
    r12_p, r12_s = fresnel_coefficients(N1, N2, theta1, theta2)
    
    beta = compute_beta(d, N1, theta1, wavelength)
    
    # Considering multiple reflections inside the film
    r_tot_p = (r01_p + r12_p * np.exp(-2j * beta)) / (1 + r01_p * r12_p * np.exp(-2j * beta))
    r_tot_s = (r01_s + r12_s * np.exp(-2j * beta)) / (1 + r01_s * r12_s * np.exp(-2j * beta))
    
    return r_tot_p, r_tot_s

# Calculate ellipsometry parameters Psi and Delta
def ellipsometry_parameters(N0, N1, N2, d, wavelength, theta):
    rp, rs = single_layer_reflectance(N0, N1, N2, d, wavelength, theta)
    psi = np.degrees(np.arctan(np.abs(rp / rs)))
    delta = np.degrees(np.angle(rp / rs))
    delta = np.mod(delta, 360)
    return psi, delta

# MSE calculation function
def mse_func(d_sio2, wavelength, psi_exp, delta_exp, sigma_psi, sigma_delta):
    psi_model = []
    delta_model = []
    
    for i in range(len(wavelength)):
        N_si = n_si[i] + 1j * k_si[i]
        N_sio2 = n_sio2[i] + 1j * k_sio2[i]
        psi, delta = ellipsometry_parameters(N_air, N_sio2, N_si, d_sio2, wavelength[i] * 1e-9, theta)
        psi_model.append(psi)
        delta_model.append(delta)
    
    psi_model = np.array(psi_model)
    delta_model = np.array(delta_model)
    
    mse = np.sqrt(
        np.sum(((psi_model - psi_exp) / sigma_psi) ** 2 + 
               ((delta_model - delta_exp) / sigma_delta) ** 2) / (2 * len(psi_exp) - 1)
    )
    
    return mse

# Apply Savitzky-Golay filter for spectral smoothing
def apply_savgol_filter(spectrum_data):
    all_spectra = []
    wavelengths = None
    for position, spectrum in spectrum_data.items():
        wavelengths, intensities = zip(*spectrum)
        wavelengths = np.array([float(w.replace('nm', '')) for w in wavelengths])
        all_spectra.append(intensities)
    
    smoothed_spectra = savgol_filter(np.array(all_spectra), window_length=11, polyorder=3, axis=1)
    
    return smoothed_spectra, wavelengths

# Read and process image data
def read_and_process_images(directory, prefix, sigma=1.3):
    print(f"Searching for files in {directory} with prefix {prefix}")
    path_pattern = os.path.join(directory, f"{prefix}_*.txt")
    files = sorted(glob.glob(path_pattern), key=lambda x: int(os.path.basename(x).replace('.txt', '').split('_')[1].replace('nm', '')))
    
    if not files:
        print(f"No files found with pattern: {path_pattern}")
        return {}, {}

    spectrum_data = {}

    for file_path in files:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            filename = os.path.basename(file_path)
            wavelength = filename.replace(prefix + '_', '').replace('.txt', '')
            content = file.readlines()
            if not content:
                print(f"No data in file {filename}")
                continue

            image_array = np.array([list(map(float, line.strip().split())) for line in content if line.strip()])
            image_array = np.rot90(image_array, k=1)  # Rotate 90 degrees counterclockwise
            image_array = np.fliplr(image_array)  # Mirror horizontally
            blurred_image = gaussian_filter(image_array, sigma=sigma)

            if not spectrum_data:
                for i in range(blurred_image.shape[0]):
                    for j in range(blurred_image.shape[1]):
                        spectrum_data[(i, j)] = []
            for i in range(blurred_image.shape[0]):
                for j in range(blurred_image.shape[1]):
                    spectrum_data[(i, j)].append((wavelength, blurred_image[i, j]))

    return spectrum_data

# Calculate average spectra for each classification
def calculate_average_spectra(classification_image, psi_images, delta_images):
    unique_classes = np.unique(classification_image)
    avg_spectra_psi = {}
    avg_spectra_delta = {}

    for cls in unique_classes:
        mask = (classification_image == cls)
        avg_spectra_psi[cls] = np.mean(psi_images[:, mask], axis=1)
        avg_spectra_delta[cls] = np.mean(delta_images[:, mask], axis=1)

    return avg_spectra_psi, avg_spectra_delta

# Plot classification image and average spectra
def plot_classification_and_spectra(classification_image, avg_spectra_delta, wavelengths_delta):
    unique_classes = np.unique(classification_image)
    
    # Display classification image
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    plt.imshow(classification_image, cmap=cmap)
    plt.colorbar(ticks=range(len(unique_classes)))
    plt.title('Classification Mask')

    # Define average spectra colors based on classification image colors
    colors = [cmap(i) for i in range(len(unique_classes))]

    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(unique_classes):
        plt.plot(wavelengths_delta, avg_spectra_delta[cls], label=f'Cluster {cls} Delta', linestyle='--', color=colors[i])
    
    plt.title('Average Spectra for Each Cluster')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Main program execution flow
root = Tk()
root.withdraw()

selected_directory = filedialog.askdirectory(title="Select image data folder")
if selected_directory:
    print(f"Selected directory: {selected_directory}")

    delta_directory = os.path.join(selected_directory, "ASCII_Delta")
    psi_directory = os.path.join(selected_directory, "ASCII_Psi")

    def load_images(directory, prefix):
        path_pattern = os.path.join(directory, f"{prefix}_*.txt")
        files = sorted(glob.glob(path_pattern), key=lambda x: int(os.path.basename(x).replace('.txt', '').split('_')[1].replace('nm', '')))
        
        images = []
        wavelengths = []
        
        for file_path in files:
            filename = os.path.basename(file_path)
            wavelength = int(filename.replace(f"{prefix}_", '').replace('.txt', '').replace('nm', ''))
            data = np.loadtxt(file_path)
            data = np.rot90(data, k=1)  # Rotate 90 degrees counterclockwise
            data = np.fliplr(data)  # Mirror horizontally
            images.append(data)
            wavelengths.append(wavelength)
        
        images = np.array(images)  # Shape: (num_wavelengths, height, width)
        wavelengths = np.array(wavelengths)
        return images, wavelengths

    # Load Psi and Delta images
    images_psi, wavelengths_psi = load_images(psi_directory, "Psi")
    images_delta, wavelengths_delta = load_images(delta_directory, "Delta")

    assert np.array_equal(wavelengths_psi, wavelengths_delta), "Psi and Delta wavelengths do not match!"

    print("Please select the classification file.")
    classification_file = filedialog.askopenfilename(title="Select classification file")
    if classification_file:
        classification_image = np.loadtxt(classification_file, dtype=int)
        print(f"Classification image loaded with shape: {classification_image.shape}")

        avg_spectra_psi, avg_spectra_delta = calculate_average_spectra(classification_image, images_psi, images_delta)
        plot_classification_and_spectra(classification_image, avg_spectra_delta, wavelengths_delta)

        initial_thickness_guess = {}
        for cls in np.unique(classification_image):
            thickness = float(input(f"Enter initial guess for thickness (nm) for class {cls}: "))
            initial_thickness_guess[cls] = thickness * 1e-9  # Convert to meters

        height, width = classification_image.shape
        thickness_map = np.zeros((height, width), dtype=float)
        mse_map = np.zeros((height, width), dtype=float)

        # Inside the loop for each pixel (i, j)
        for i in range(height):
            for j in range(width):
                cls = classification_image[i, j]
                d_guess = initial_thickness_guess[cls]

                psi_spectrum = images_psi[:, i, j]
                delta_spectrum = images_delta[:, i, j]

                if psi_spectrum.size == 0 or delta_spectrum.size == 0:
                    thickness_map[i, j] = np.nan
                    mse_map[i, j] = np.nan
                    continue

                wavelengths_m = wavelengths_psi * 1e-9

                sigma_psi = 0.1
                sigma_delta = 0.1

                thickness_range = np.arange(max(d_guess - 5e-9, 0), d_guess + 5e-9, 8e-10)
                mse_values = []

                for d in thickness_range:
                    mse = mse_func(d, wavelengths_m, psi_spectrum, delta_spectrum, sigma_psi, sigma_delta)
                    mse_values.append(mse)

                # Cubic polynomial fitting to obtain more accurate thickness
                p = np.polyfit(thickness_range, mse_values, 3)
                
                # Evaluate the polynomial over a finer grid to find the minimum
                fine_thickness_range = np.linspace(thickness_range.min(), thickness_range.max(), 1000)
                fine_mse_values = np.polyval(p, fine_thickness_range)
                fine_thickness = fine_thickness_range[np.argmin(fine_mse_values)]

                best_thickness = fine_thickness if thickness_range.min() <= fine_thickness <= thickness_range.max() else thickness_range[np.argmin(mse_values)]
                thickness_map[i, j] = best_thickness

                # Record minimum MSE in MSE map
                min_mse = np.min(mse_values)
                mse_map[i, j] = min_mse

                # Print the fitted thickness and corresponding MSE value
                print(f"Pixel ({i}, {j}) fitted thickness: {best_thickness * 1e9:.2f} nm, MSE: {min_mse:.4f}")


        # Normalize MSE map, making the minimum value 1
        mse_map_norm = mse_map / np.nanmin(mse_map)

        # Rotate and mirror thickness and MSE maps
        thickness_map_rotated = np.fliplr(np.rot90(thickness_map, k=2))
        mse_map_norm_rotated = np.fliplr(np.rot90(mse_map_norm, k=2))

        # Remove the first row from both maps
        thickness_map_rotated = thickness_map_rotated[1:, :]
        mse_map_norm_rotated = mse_map_norm_rotated[1:, :]

        # Plot 2D film thickness distribution
        plt.figure(figsize=(8, 6))
        img = plt.imshow(thickness_map_rotated * 1e9, cmap='viridis', origin='lower')
        cbar = plt.colorbar(img, format='%.1f', pad=0.05)  # Add padding to avoid overlap with color bar
        
        # Control the number of ticks to avoid overlap
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Dynamically generate ticks to ensure they are not too close to the edges
        min_thickness = thickness_map_rotated.min() * 1e9
        max_thickness = thickness_map_rotated.max() * 1e9
        thickness_ticks = np.linspace(min_thickness, max_thickness, 5)
        cbar.set_ticks(thickness_ticks)
        cbar.set_ticklabels([f'{tick:.1f}' for tick in thickness_ticks])  # 保留一位小數

        plt.title('2D Film Thickness Distribution (nm)')
        plt.show()

        # Plot normalized MSE map
        plt.figure(figsize=(8, 6))
        img = plt.imshow(mse_map_norm_rotated, cmap='viridis', origin='lower')
        cbar = plt.colorbar(img, format='%.2f', pad=0.05)  # Add padding to avoid overlap with color bar

        # Control the number of ticks to avoid overlap
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Dynamically generate ticks to ensure they are not too close to the edges
        min_val = mse_map_norm_rotated.min()
        max_val = mse_map_norm_rotated.max()
        ticks = np.linspace(min_val, max_val, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])  # 保留兩位小數

        plt.title('Normalized MSE Map')
        plt.show()


        # Plot 3D film thickness distribution
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(width), np.arange(height-1))  # Adjust for row removal
        ax.plot_surface(X, Y, thickness_map_rotated * 1e9, cmap='viridis')
        ax.set_xlabel('X axis (pixels)')
        ax.set_ylabel('Y axis (pixels)')
        ax.set_zlabel('Thickness (nm)')
        ax.set_title('3D Film Thickness Distribution')

        # Show all figures at once, after all plotting commands
        plt.show()

        # Save thickness and MSE maps as text files
        np.savetxt(os.path.join(selected_directory, "thickness_map.txt"), thickness_map_rotated * 1e9, fmt="%.6f")
        np.savetxt(os.path.join(selected_directory, "mse_map_norm.txt"), mse_map_norm_rotated, fmt="%.6f")

    else:
        print("No classification file selected. Exiting...")
else:
    print("No directory selected. Exiting...")

root.destroy()
