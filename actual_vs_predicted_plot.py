"""
This script plots the distribution of the labels
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data

file_path = "stacked_matrices_OF_line14_Z3_2M.h5"
file_path_NF = "stacked_matrices_NF_line14_Z3_2M.h5"

file_path_line13 = "stacked_matrices_OF_Z3_2M.h5"
file_path_NF_line13 = "stacked_matrices_NF_Z3_2M.h5"

with h5py.File(file_path, 'r') as f:
    stacked_matrices = f['stacked_matrices'][:]
with h5py.File(file_path_NF, 'r') as f:
    stacked_matrices_NF = f['stacked_matrices_NF'][:]

with h5py.File(file_path_line13, 'r') as f:
    stacked_matrices_line13 = f['stacked_matrices'][:]
with h5py.File(file_path_NF_line13, 'r') as f:
    stacked_matrices_NF_line13 = f['stacked_matrices_NF'][:]
print(stacked_matrices.shape)
print(stacked_matrices_NF.shape)
print("---------------")
OF = stacked_matrices  # Operational features
NF_3D = stacked_matrices_NF  # Non-operational features
NF_2D = NF_3D[:, 0, :]  # Extract the first row from each 3x3 matrix
NF = NF_2D

OF_line13 = stacked_matrices_line13  # Operational features
NF_3D_line13 = stacked_matrices_NF_line13  # Non-operational features
NF_2D_line13 = NF_3D_line13[:, 0, :]  # Extract the first row from each 3x3 matrix
NF_line13 = NF_2D_line13

# Assuming Z, H, start, end, and delay_changes are defined as in your previous script
Z, H = 3, 3
start = Z * 4
end = Z * 4 + Z
arrival_delays = OF[:, :, start:end]
arrival_delays_line13 = OF_line13[:, :, start:end]


labels = []
for i in range(OF.shape[0]):
    sub_matrix_idx = (i + 1) % OF.shape[0]
    label = arrival_delays[sub_matrix_idx, -1, -1]
    labels.append([label])
labels = np.array(labels)

labels_line13 = []
for i in range(OF_line13.shape[0]):
    sub_matrix_idx_line13 = (i + 1) % OF_line13.shape[0]
    label_line13 = arrival_delays_line13[sub_matrix_idx_line13, -1, -1]
    labels_line13.append([label_line13])
labels_line13 = np.array(labels_line13)

labels_test_flat = labels.flatten()
labels_test_flat_line13 = labels_line13.flatten()

counts, bin_edges = np.histogram(labels_test_flat_line13, bins=60)

# Plot the distributions
plt.figure(figsize=(10, 6))

# Plot distribution for labels_test_flat_line13 with the same bin intervals
sns.histplot(labels_test_flat_line13, kde=True, bins=bin_edges, color='orange', label='line 13')

# Plot distribution for labels_test_flat with the same bin intervals
sns.histplot(labels_test_flat, kde=True, bins=bin_edges, color='blue', label='line 14')

plt.xlim(0, 400)  # Set the x-axis range to 0 to 400
plt.title('Distribution of Arrival Delays Target Values')
plt.xlabel('Arrival Delay (seconds)')
plt.ylabel('Frequency')

# Add the legendplot_map
plt.legend()

# Show the plot
plt.show()

# Print the bin edges and counts for labels_test_flat_line13
print("Bin edges:", bin_edges)
print("Counts per bin:", counts)

# Optional: Print ranges and their frequencies for labels_test_flat_line13
for i in range(len(counts)):
    print(f"Range {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {counts[i]}")

# Calculate the mean value of labels_test_flat
mean_value = np.mean(labels_test_flat)
print(f"Mean value: {mean_value}")