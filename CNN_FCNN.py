import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import numpy as np
import shap

file_path = "stacked_matrices_OF_line14_1M.h5"
file_path_NF = "stacked_matrices_NF_line14_1M.h5"


with h5py.File(file_path, 'r') as f:
    stacked_matrices = f['stacked_matrices'][:]
with h5py.File(file_path_NF, 'r') as f:
    stacked_matrices_NF = f['stacked_matrices_NF'][:]

OF = stacked_matrices  # Operational features
NF_3D = stacked_matrices_NF  # Non-operational features
NF_2D = NF_3D[:, 0, :]  # Extract the first row from each 3x3 matrix
NF = NF_2D

# Assuming Z, H, start, end, and delay_changes are defined as in your previous script
Z, H = 3, 3
start = Z * 4
end = Z * 4 + Z
arrival_delays = OF[:, :, start:end]
delay_changes = np.diff(arrival_delays, axis=2)


labels = []
for i in range(OF.shape[0]):
    sub_matrix_idx = (i + 1) % OF.shape[0]
    label = arrival_delays[sub_matrix_idx, -1, -1]
    labels.append([label])
print(labels[1005:1008])
labels = np.array(labels)
print(labels.shape)

labels_test_flat = labels.flatten()

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.histplot(labels_test_flat, kde=True, bins=30)
plt.xlim(0, 400)  # Set the x-axis range to 0 to 800
plt.title('Distribution of Labels Test Values')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

counts, bin_edges = np.histogram(labels, bins=30)

# Print the frequency for each bin
print("Bin edges:", bin_edges)
print("Counts per bin:", counts)

# Optional: Print ranges and their frequencies
for i in range(len(counts)):
    print(f"Range {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {counts[i]}")

# Calculate the mean value
mean_value = np.mean(labels_test_flat)
print(f"Mean value: {mean_value}")
# Flatten the delay changes to use as regression targets
#labels = delay_changes.reshape(delay_changes.shape[0], -1)

# Expand the OF matrix to add a channel dimension
OF_expanded = np.expand_dims(OF, axis=-1)

# Define the CNN model for operational features
input_cnn = Input(shape=(OF_expanded.shape[1], OF_expanded.shape[2], 1))
x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_cnn)
x = MaxPooling2D((1, 1), padding='same')(x)

x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 1), padding='same')(x)

x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)

# Define the FCNN model for non-operational features
input_fcnn = Input(shape=(NF.shape[1],))
y = Dense(64, activation='relu')(input_fcnn)
y = Dense(64, activation='relu')(y)
y = Dense(64, activation='relu')(y)
y = Dense(64, activation='relu')(y)

# Merging the outputs of CNN and FCNN
merged = Concatenate()([x, y])

# Further processing after merging
z = Dense(256, activation='relu')(merged)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
output = Dense(labels.shape[1], activation='linear')(z)

# Create the model
model = Model(inputs=[input_cnn, input_fcnn], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Split data into training and testing sets
X_train_op, X_test_op, X_train_nonop, X_test_nonop, y_train, y_test = train_test_split(OF_expanded, NF, labels, test_size=0.2, random_state=42)
print(X_test_op.shape)
print(X_test_nonop.shape)
print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = model.fit([X_train_op, X_train_nonop], y_train, epochs=3, batch_size=128, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# Save the model if needed
#model.save('model_name')
#print("Model saved to disk")

# Evaluate the model
test_loss, test_mae = model.evaluate([X_test_op, X_test_nonop], y_test)
print(f'Test MAE: {test_mae:.4f}')

# Plot learning curves
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Train')
plt.plot(history.history['val_mean_absolute_error'], label='Validation')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Make predictions on the test set
predictions = model.predict([X_test_op, X_test_nonop])

# Flatten the arrays for plotting
y_test_flat = y_test.flatten()
predictions_flat = predictions.flatten()

# Plot the actual vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_flat, y=predictions_flat, s=50, label='Data Points')

plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'k--', label='Ideal Line (y=x)')

# Add labels, title, and legend
plt.xlabel('Actual Delay (seconds)')
plt.ylabel('Predicted Delay (seconds)')
plt.title('Actual vs. Predicted Delays')
plt.legend()
plt.show()

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
epsilon = np.finfo(float).eps  # Small value to prevent division by zero
mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, epsilon))) * 100
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(f'R-squared: {r2}')
print("--------------------------------")


# Initialize the SHAP DeepExplainer
# Initialize SHAP DeepExplainer
data_ponts = 200

# Initialize JavaScript for SHAP plots
shap.initjs()

# Calculate SHAP values using DeepExplainer
explainer = shap.DeepExplainer(model, [X_train_op[:data_ponts], X_train_nonop[:data_ponts]])
shap_values = explainer.shap_values([X_test_op[:data_ponts], X_test_nonop[:data_ponts]])

# Print shapes to debug
print(f"Shape of shap_values[0]: {np.array(shap_values[0]).shape}")
print(f"Shape of X_test_op[:100]: {X_test_op[:data_ponts].shape}")

# Reshape SHAP values
shap_values_reshaped = shap_values[0].reshape((shap_values[0].shape[0], shap_values[0].shape[1], -1))

# Aggregate SHAP values by summing over the trains axis (axis=1)
shap_values_aggregated = np.sum(shap_values_reshaped, axis=1)

# Similarly, aggregate the input features for the SHAP plot
X_test_op_reshaped = X_test_op[:data_ponts].reshape((X_test_op[:data_ponts].shape[0], X_test_op[:data_ponts].shape[1], -1))
X_test_op_aggregated = np.mean(X_test_op_reshaped, axis=1)

# Define the number of unique features
num_unique_features = shap_values_aggregated.shape[1]  # Adjust this based on the actual number of unique features

# Create custom feature names for the aggregated features
name_list_Z3 = ["T2", "T1", "T0", "W2", "W1", "W0", "R2", "R1", "R0", "D2", "D1", "D0", "Y2", "Y1", "Y0", "T'2", "T´1",
                "T'0", "W'2", "W'1", "W´0", "R´2", "R'1", "R'0", "S'2", "S'1", "S'0"]
#custom_feature_names = [f'Feature_{i}' for i in range(num_unique_features)]
custom_feature_names = [f'{name_list_Z3[i]}' for i in range(num_unique_features)]


# Generate the SHAP summary plot
shap.summary_plot(shap_values_aggregated, X_test_op_aggregated, feature_names=custom_feature_names, max_display=num_unique_features)

feature_index = 11  # For example

# Plot the SHAP dependence plot
shap.dependence_plot(feature_index, shap_values_aggregated, X_test_op_aggregated, feature_names=custom_feature_names)

# Ensure expected_value is a scalar (for single output) or select the appropriate one (for multi-output models)
expected_value = np.array(explainer.expected_value)

# If your model has multiple outputs, you might need to select the appropriate expected value
if expected_value.ndim > 0:
    expected_value = expected_value[0]  # Adjust index based on your setup

# Plot the decision plot
shap.decision_plot(expected_value, shap_values_aggregated, X_test_op_aggregated, feature_names=custom_feature_names)

# Ensure all data are numpy arrays
shap_values_aggregated = np.array(shap_values_aggregated)
X_test_op_aggregated = np.array(X_test_op_aggregated)

# Choose an instance to plot
instance_index = 11
# Create the force plot
force_plot = shap.force_plot(expected_value, shap_values_aggregated, X_test_op_aggregated, feature_names=custom_feature_names)
# Save the force plot as an HTML file
shap.save_html('force_plot.html', force_plot)

# At this point, you can open 'force_plot.html' in your browser to view the plot.

shap.waterfall_plot(shap.Explanation(values=shap_values_aggregated[instance_index], base_values=expected_value, data=X_test_op_aggregated[instance_index], feature_names=custom_feature_names))

shap.summary_plot(shap_values_aggregated, X_test_op_aggregated, plot_type="bar", feature_names=custom_feature_names)

data_points = 200  # Number of data points to use for SHAP explanations

# Initialize JavaScript for SHAP plots
shap.initjs()

# Calculate SHAP values using DeepExplainer for non-operational data
explainer = shap.DeepExplainer(model, [X_train_op[:data_points], X_train_nonop[:data_points]])
shap_values = explainer.shap_values([X_test_op[:data_points], X_test_nonop[:data_points]])

# Extract SHAP values for non-operational data
shap_values_nonop = shap_values[1].squeeze()  # Remove the extra dimension

# Print shapes to debug
print(f"Shape of shap_values_nonop: {shap_values_nonop.shape}")
print(f"Shape of X_test_nonop[:data_points]: {X_test_nonop[:data_points].shape}")

# Ensure the number of features and feature names align
num_unique_features_nonop = shap_values_nonop.shape[1]
custom_feature_names_nonop = ["L2", "L1", "L0", "M2", "M1", "M0"]

# Check if the feature names list matches the number of features
assert len(custom_feature_names_nonop) == num_unique_features_nonop, \
    "Number of feature names does not match the number of features."

# Generate SHAP summary plot for non-operational data
shap.summary_plot(shap_values_nonop, X_test_nonop[:data_points], feature_names=custom_feature_names_nonop, max_display=num_unique_features_nonop)

# Generate dependence plot for a specific feature index (example: feature_index = 11)
feature_index_nonop = 0
shap.dependence_plot(feature_index_nonop, shap_values_nonop, X_test_nonop[:data_points], feature_names=custom_feature_names_nonop)

# Ensure expected_value is a scalar (for single output) or select the appropriate one (for multi-output models)
expected_value = np.array(explainer.expected_value)

# If your model has multiple outputs, you might need to select the appropriate expected value
if expected_value.ndim > 0:
    expected_value = expected_value[0]  # Adjust index based on your setup

# Plot the decision plot for non-operational data
shap.decision_plot(expected_value, shap_values_nonop, X_test_nonop[:data_points], feature_names=custom_feature_names_nonop)

# Choose an instance to plot
instance_index = 0

# Create the force plot for a chosen instance
force_plot_nonop = shap.force_plot(expected_value, shap_values_nonop[instance_index], X_test_nonop[instance_index], feature_names=custom_feature_names_nonop)

# Save the force plot as an HTML file
shap.save_html('force_plot_nonop.html', force_plot_nonop)

# Generate waterfall plot for non-operational data
shap.waterfall_plot(shap.Explanation(values=shap_values_nonop[instance_index], base_values=expected_value, data=X_test_nonop[instance_index], feature_names=custom_feature_names_nonop))

# Generate bar summary plot for non-operational data
shap.summary_plot(shap_values_nonop, X_test_nonop[:data_points], plot_type="bar", feature_names=custom_feature_names_nonop)