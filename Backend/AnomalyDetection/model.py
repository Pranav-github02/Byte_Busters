import ipaddress
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
import pickle

# Set the file path
# Update the file path without double quotes
file_path = '..........'

# Create an instance of the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust the contamination parameter as needed

chunk_size = 10000

# Read the dataset
data = pd.read_csv(file_path)

# Select the features for anomaly detection
features = ['Login Timestamp', 'User ID', 'IP Address', 'Login Successful']
categorical_features = ['Country', 'Region', 'City', 'Browser Name and Version', 'Device Type']

# Prepare an empty DataFrame to store the processed chunks
processed_data = pd.DataFrame()

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Extract the selected features from the chunk
    X = chunk[features].copy()

    # Convert 'Login Timestamp' to UNIX timestamp
    X['Login Timestamp'] = pd.to_datetime(X['Login Timestamp'])
    X['Login Timestamp'] = X['Login Timestamp'].apply(lambda x: datetime.timestamp(x))

    # Convert 'IP Address' to integer representation
    X['IP Address'] = X['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))

    # Append the processed chunk to the overall processed data
    processed_data = pd.concat([processed_data, X], ignore_index=True)

# Fit the model to the processed data
model.fit(processed_data)

# Prepare the data for anomaly prediction
data_copy = data.copy()
data_copy['Login Timestamp'] = pd.to_datetime(data_copy['Login Timestamp'])
data_copy['Login Timestamp'] = data_copy['Login Timestamp'].apply(lambda x: datetime.timestamp(x))
data_copy['IP Address'] = data_copy['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))

# Encode categorical columns with label encoding
for feature in categorical_features:
    data_copy[feature] = pd.factorize(data_copy[feature])[0]

# Predict the anomalies (1 for normal, -1 for anomalies)
predictions = model.predict(data_copy[features])

# Add the anomaly predictions to the original dataset
data['Anomaly'] = predictions

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


