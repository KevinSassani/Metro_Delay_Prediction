import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
file_path = "stacked_matrices_OF_line14_1M.h5"
file_path_NF = "stacked_matrices_NF_line14_1M.h5"

with h5py.File(file_path, 'r') as f:
    stacked_matrices = f['stacked_matrices'][:] #Change 'stacked_matrices' -> 'stacked_matrices_Q' if Q-learning
with h5py.File(file_path_NF, 'r') as f:
    stacked_matrices_NF = f['stacked_matrices_NF'][:] #Change 'stacked_matrices_NF' -> 'stacked_matrices_NF_Q' if Q-learning

H = 3 # Number of consecutive trains to consider for feature extraction
Z = 3 # Number of stations or sections the input data is derived from

# Operational features (OF) and Non-operational features (NF) from the input data
OF = stacked_matrices
NF = stacked_matrices_NF

# Prepare OFi (input data for LSTM) by creating sliding windows of operational features
OFi = []
for i in range(OF.shape[0]):
    if i < H:
        # For the first H samples, pad the window with zeros to maintain consistent window size
        padded_window = np.zeros((H - i, OF.shape[1] * OF.shape[2]))
        # Extract the available part of the window and reshape it into a 1D array
        full_window = OF[:i].reshape(-1)
        # Combine the padded zeros with the actual data to form a complete window
        window = np.concatenate((padded_window.flatten(), full_window))
    else:
        # For samples beyond the first H, extract the full sliding window without padding
        window = OF[i - H:i].reshape(-1)
    # Append the constructed window to the OFi list
    OFi.append(window)

# Convert the list OFi to a NumPy array for further processing
OFi = np.array(OFi)

# Reshape OFi to fit the LSTM input requirements (samples, time steps, features)
OFi = OFi.reshape((OFi.shape[0], H, OF.shape[1] * OF.shape[2]))

# Prepare OFp (another version of input data for LSTM) in a similar manner
OFp = []
for i in range(OF.shape[0]):
    if i < H:
        # For the first H samples, pad the window with zeros to maintain consistent window size
        padded_window = np.zeros((H - i, OF.shape[1] * OF.shape[2]))
        # Extract the available part of the window and reshape it into a 1D array
        full_window = OF[:i].reshape(-1)
        # Combine the padded zeros with the actual data to form a complete window
        window = np.concatenate((padded_window.flatten(), full_window))
    else:
        # For samples beyond the first H, extract the full sliding window without padding
        window = OF[i - H:i, :, :].reshape(-1)
    # Append the constructed window to the OFp list
    OFp.append(window)

# Convert the list OFp to a NumPy array for further processing
OFp = np.array(OFp)

# Reshape OFp to fit the LSTM input requirements (samples, time steps, features)
OFp = OFp.reshape((OFp.shape[0], H, OF.shape[1] * OF.shape[2]))

# Extract labels (arrival delays) from the operational features
start = Z * 4  # Starting index for the arrival delays within OF
end = Z * 4 + Z  # Ending index for the arrival delays within OF
arrival_delays = OF[:, :, start:end]

# Initialize an empty list to store the labels
labels = []
for i in range(OF.shape[0]):
    # Get the index of the subsequent sub-matrix (wrapping around using modulo)
    sub_matrix_idx = (i + 1) % OF.shape[0]

    # Extract the arrival delay label for the last station of the next sample
    label = arrival_delays[sub_matrix_idx, -1, -1]
    labels.append([label])

# Convert the list of labels into a NumPy array
labels = np.array(labels)

# Ensure the number of samples in labels matches that of OFi and OFp
if labels.shape[0] != OFi.shape[0]:
    min_samples = min(labels.shape[0], OFi.shape[0])

    # Truncate both labels and OFi/OFp to have the same number of samples
    labels = labels[:min_samples]
    OFi = OFi[:min_samples]
    OFp = OFp[:min_samples]

# Split data into training and testing sets
X_train_ofi, X_test_ofi, X_train_ofp, X_test_ofp, X_train_nf, X_test_nf, y_train, y_test = train_test_split(
    OFi, OFp, NF, labels, test_size=0.2, random_state=42)

# Define the LSTM model for operational features (train interactions)
input_lstm1 = Input(shape=(H, OF.shape[1] * OF.shape[2]))
x1 = LSTM(128, return_sequences=True)(input_lstm1)
x1 = LSTM(128, return_sequences=False)(x1)

# Define the LSTM model for operational features (station interactions)
input_lstm2 = Input(shape=(H, OF.shape[1] * OF.shape[2]))
x2 = LSTM(128, return_sequences=True)(input_lstm2)
x2 = LSTM(128, return_sequences=False)(x2)

# Define the LSTM model for non-operational features
input_lstm3 = Input(shape=(NF.shape[1], NF.shape[2]))
y = LSTM(128, return_sequences=True)(input_lstm3)
y = LSTM(128, return_sequences=False)(y)

# Merging the outputs of the LSTMs
merged = Concatenate()([x1, x2, y])

# Further processing after merging
z = Dense(256, activation='relu')(merged)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
z = Dense(256, activation='relu')(z)
output = Dense(labels.shape[1], activation='linear')(z)

# Create the model
model = Model(inputs=[input_lstm1, input_lstm2, input_lstm3], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Train the model
history = model.fit([X_train_ofi, X_train_ofp, X_train_nf], y_train, epochs=100, batch_size=64, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])
#model.save('my_model_LSTM_NF.h5')
# Evaluate the model
test_loss, test_mae = model.evaluate([X_test_ofi, X_test_ofp, X_test_nf], y_test)
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

# Make predictions on the test data
predictions = model.predict([X_test_ofi, X_test_ofp, X_test_nf])

# Flatten the arrays
y_test_flat = y_test.flatten()
predictions_flat = predictions.flatten()

# Plot the actual vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_flat, y=predictions_flat, s=50, label='Data Points')

# Plot the ideal line
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
