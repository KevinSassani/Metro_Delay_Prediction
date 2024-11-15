import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from tqdm import tqdm
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

# Load the data
file_path = "stacked_matrices_OF_line14_1M.h5"
file_path_NF = "stacked_matrices_NF_line14_1M.h5"

with h5py.File(file_path, 'r') as f:
    stacked_matrices = f['stacked_matrices_Q'][:]
with h5py.File(file_path_NF, 'r') as f:
    stacked_matrices_NF = f['stacked_matrices_NF_Q'][:]

# Number of consecutive trains to consider
H = 3
# Ensure the labels (arrival delays) have the same number of samples as OFi
Z = 3  # Number of stations/sections from which the input data is derived
start = Z * 4  # Calculate the starting index for the arrival delays within OF
end = Z * 4 + Z  # Calculate the ending index for the arrival delays within OF

# Operational Features (OFi) or (OF) and Normalized Features (NF) from the input data
OFi = stacked_matrices
OF = OFi
NF = stacked_matrices_NF

# Reshape OFi for LSTM input by transposing the axes to (samples, stations, features)
OFp = np.transpose(OFi, (0, 2, 1))

# Extract the arrival delays as the target labels
arrival_delays = OF[:, :, start:end]

# Initialize an empty list to store the labels
labels = []
for i in range(OF.shape[0]):
    # Get the index of the subsequent sub-matrix (wrapping around using modulo)
    sub_matrix_idx = (i + 1) % OF.shape[0]
    # Extract the arrival delay label for the last station of the next sample
    label = arrival_delays[sub_matrix_idx, -1, -1]
    labels.append([label])

labels = np.array(labels)

# Ensure the number of samples in labels matches that of OFi
if labels.shape[0] != OFi.shape[0]:
    min_samples = min(labels.shape[0], OFi.shape[0])
    # Truncate both labels and OFi/OFp to have the same number of samples
    labels = labels[:min_samples]
    OFi = OFi[:min_samples]
    OFp = OFp[:min_samples]

# Split data into training and testing sets
X_train_ofi, X_test_ofi, X_train_ofp, X_test_ofp, X_train_fcnn, X_test_fcnn, y_train, y_test = train_test_split(
    OFi, OFp, NF, labels, test_size=0.2, random_state=42)

# Define the first LSTM component for operational features (train interactions)
input_lstm1 = Input(shape=(H, OF.shape[2]))
x1 = LSTM(128, return_sequences=True)(input_lstm1)
x1 = LSTM(128, return_sequences=False)(x1)

# Define the second LSTM component for operational features (station interactions)
input_lstm2 = Input(shape=(OF.shape[2], H))
x2 = LSTM(128, return_sequences=True)(input_lstm2)
x2 = LSTM(128, return_sequences=False)(x2)

# Define the FCNN component for non-operational features
input_fcnn = Input(shape=(X_train_fcnn.shape[1],))
y = Dense(64, activation='relu')(input_fcnn)
y = Dense(64, activation='relu')(y)
y = Dense(64, activation='relu')(y)
y = Dense(64, activation='relu')(y)

# Merging the outputs of LSTM and FCNN
merged = Concatenate()([x1, x2, y])

# Define the FCNN component after merging
z = Dense(256, activation='relu')(merged)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
output = Dense(labels.shape[1], activation='linear')(z)

# Create the model
model = Model(inputs=[input_lstm1, input_lstm2, input_fcnn], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Train the model
history = model.fit([X_train_ofi, X_train_ofp, X_train_fcnn], y_train, epochs=100, batch_size=128, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# Save the model if needed
#model.save('model_name')
#print("Model saved to disk")

# Evaluate the model
test_loss, test_mae = model.evaluate([X_test_ofi, X_test_ofp, X_test_fcnn], y_test)
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
predictions = model.predict([X_test_ofi, X_test_ofp, X_test_fcnn])

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
print("--------------------------")

# Calculate residuals
residuals = y_test_flat - predictions_flat

# Plot the residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=60)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(predictions_flat, residuals, alpha=0.5)
plt.hlines(0, min(predictions_flat), max(predictions_flat), colors='r', linestyles='dashed')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# QQ plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

dw_statistic = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_statistic}')

# Add constant to the predictor for statsmodels
X_test_with_const = sm.add_constant(predictions_flat)

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(residuals, X_test_with_const)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_flat, y=predictions_flat, s=50, label='Data Points')
plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'k--', label='Ideal Line (y=x)')
plt.xlabel('Actual Delay (seconds)')
plt.ylabel('Predicted Delay (seconds)')
plt.title('Actual vs. Predicted Delays')
plt.legend()
plt.show()

# Calculate residuals
residuals = y_test_flat - predictions_flat

# Plot the residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=60)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(predictions_flat, residuals, alpha=0.5)
plt.hlines(0, min(predictions_flat), max(predictions_flat), colors='r', linestyles='dashed')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# QQ plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

dw_statistic = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_statistic}')

# Add constant to the predictor for statsmodels
X_test_with_const = sm.add_constant(predictions_flat)

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(residuals, X_test_with_const)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_flat, y=predictions_flat, s=50, label='Data Points')
plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'k--', label='Ideal Line (y=x)')
plt.xlabel('Actual Delay (seconds)')
plt.ylabel('Predicted Delay (seconds)')
plt.title('Actual vs. Predicted Delays')
plt.legend()
plt.show()

