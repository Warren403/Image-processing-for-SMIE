import numpy as np
import matplotlib.pyplot as plt

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

# Define MSE
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
        np.sum(((psi_model - psi_exp) / sigma_psi) ** 2 + ((delta_model - delta_exp) / sigma_delta) ** 2) / (2 * len(psi_exp) - 1)
    )
    
    return mse

# set the path of the file, the test data can be exp data
si_file_path = r'E:\image processing for elliposmetry\Si.txt'
sio2_file_path = r'E:\image processing for elliposmetry\SiO2.txt'
test_data_file_path = r'E:\image processing for elliposmetry\AOI_40_SE dataC1.txt'

# load optical constant of Si and SiO2 
wavelength_si, n_si, k_si = np.loadtxt(si_file_path, unpack=True)
wavelength_sio2, n_sio2, k_sio2 = np.loadtxt(sio2_file_path, unpack=True)

# load exp data
exp_data = np.loadtxt(test_data_file_path, unpack=True)
wavelength_exp = exp_data[0]
psi_exp = exp_data[1]
delta_exp = exp_data[2]
sigma_psi = np.full_like(psi_exp, 0.1)  # set sigma=0.1
sigma_delta = np.full_like(delta_exp, 0.1)  # same as line 77

# input the thickness initial guess (the unit is nm)
initial_guess = float(input("Enter the initial guess for thickness (in nm): ")) * 1e-9

# set the scanning thickness range
thickness_range = np.arange(initial_guess - 4e-9, initial_guess + 4e-9, 0.01e-9)

# define the air refractive index and AOI
N_air = 1.0 + 0j
theta = np.radians(40)  # 入射角度40度

# save the mse for evrey scanned thickness
mse_values = []

for d in thickness_range:
    mse = mse_func(d, wavelength_exp, psi_exp, delta_exp, sigma_psi, sigma_delta)
    mse_values.append(mse)

# find the minimum MSE
best_thickness_index = np.argmin(mse_values)
best_thickness = thickness_range[best_thickness_index]
print(f"Best SiO2 thickness: {best_thickness * 1e9:.2f} nm")

# use the best thickness to calculate the psi 和 delta
fitted_psi_values = []
fitted_delta_values = []

for i in range(len(wavelength_si)):
    N_si = n_si[i] + 1j * k_si[i]
    N_sio2 = n_sio2[i] + 1j * k_sio2[i]
    psi, delta = ellipsometry_parameters(N_air, N_sio2, N_si, best_thickness, wavelength_si[i] * 1e-9, theta)
    fitted_psi_values.append(psi)
    fitted_delta_values.append(delta)

# plot the fitting result---
fig, ax1 = plt.subplots()  # Adjust figure size

# First y-axis (Psi)
color = 'tab:red'
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Ψ in degrees', color=color)
line1, = ax1.plot(wavelength_si, fitted_psi_values, color=color, linestyle='-', linewidth=2, label='Thm Psi')  
line_exp_psi, = ax1.plot(wavelength_exp, psi_exp, color='black', marker='o', markersize=8, 
                         markerfacecolor='none', linestyle='-', label='Exp Psi')  
ax1.tick_params(axis='y', labelcolor=color)


# Second y-axis (Delta)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Δ in degrees', color=color)  
line2, = ax2.plot(wavelength_si, fitted_delta_values, color=color, linestyle='-', linewidth=2, label='Thm Delta') 
line_exp_delta, = ax2.plot(wavelength_exp, delta_exp, color='black', marker='^', markersize=8, 
                           markerfacecolor='none', linestyle='-', label='Exp Delta')  
ax2.tick_params(axis='y', labelcolor=color)


# Adjust the layout
fig.tight_layout()  

# Title
plt.title('Experimental and Generated Data at AOI=40°')

# Adjust legend placement inside the plot area
lines = [line1, line_exp_psi, line2, line_exp_delta]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='upper right', frameon=True, fontsize=9, bbox_to_anchor=(0.95, 0.95))  # Moved legend inside

# Display the plot
plt.show()


# plot MSE vs thickness, noting the minimum
plt.figure()

plt.scatter(thickness_range * 1e9, mse_values, color='blue', s=10, label='Data Points')  

plt.scatter([best_thickness * 1e9], [mse_values[best_thickness_index]], color='red', s=50, label='Minimum MSE Point')  

plt.text(best_thickness * 1e9, mse_values[best_thickness_index] + 0.7,  
         f'({best_thickness * 1e9:.2f} nm, {mse_values[best_thickness_index]:.2f})', 
         fontsize=14, color='red', weight='bold', ha='center', va='bottom')  

plt.xlabel('Thickness (nm)')
plt.ylabel('MSE')
plt.title('MSE vs Thickness')

plt.legend()
plt.show()


