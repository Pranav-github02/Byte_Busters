import ipaddress

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
import pickle
import pyarrow as pa
import pyarrow.parquet as pq




file_path = 'C:/Users/ADITYA GARG/Downloads/Chitkara_Anomaly_Detection (1)/Chitkara_Anomaly_Detection/Login_Data.csv'  # Update the file path without double quotes

# Read the dataset
df = pd.read_csv(file_path)
table = pa.Table.from_pandas(df)
pq.write_table(table, 'output.parquet')

data = pd.read_parquet('output.parquet')

# Convert 'Login Timestamp' to UNIX timestamp
data['Login Timestamp'] = pd.to_datetime(data['Login Timestamp'])
data['Login Timestamp'] = data['Login Timestamp'].apply(lambda x: datetime.timestamp(x))
data['IP Address'] = data['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))
# Select the features for anomaly detection (excluding 'IP Address')
features = ['Login Timestamp', 'User ID', 'IP Address', 'Login Successful']

# Convert categorical columns to one-hot encoding
categorical_features = ['Country', 'Region', 'City', 'Browser Name and Version', 'Device Type']
data = pd.get_dummies(data, columns=categorical_features)

# Extract the selected features from the dataset
X = data[features]
# Create an instance of the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust the contamination parameter as needed

# Fit the model to the data
model.fit(X)

# Predict the anomalies (1 for normal, -1 for anomalies)
predictions = model.predict(X)

# Add the anomaly predictions to the original dataset
data['Anomaly'] = predictions

# Assuming 'model' is your trained machine learning model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)




