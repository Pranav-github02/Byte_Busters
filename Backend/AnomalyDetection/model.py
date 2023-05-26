from datetime import datetime
import pandas as pd
import numpy as np
import geoip2.database
from keras.models import Sequential
import pickle
import ipaddress
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler



def get_geolocation(ip_address):
    reader = geoip2.database.Reader('C:/Users/anike/Downloads/GeoLite2-City_20230523/GeoLite2-City_20230523/GeoLite2-City.mmdb')
    try:
        response = reader.city(ip_address)
        country_code = response.country.iso_code
        return country_code
    except geoip2.errors.AddressNotFoundError:
        return None
    finally:
        reader.close()


def check_compatibility(browser, device_type):
    if browser.startswith('Firefox'):
        return device_type
    elif browser.startswith('Chrome Mobile') and device_type in ['mobile', 'tablet']:
        return device_type
    elif browser.startswith('Android') and device_type in ['mobile', 'tablet']:
        return device_type
    elif browser.startswith('Chrome Mobile WebView') and device_type == 'mobile':
        return device_type
    elif browser.startswith('Chrome') and device_type == 'desktop':
        return device_type
    elif browser.startswith('Opera') and device_type in ['desktop', 'mobile', 'tablet']:
        return device_type
    elif browser.startswith('MiuiBrowser') and device_type == 'mobile':
        return device_type
    elif browser.startswith('UC Browser') and device_type in ['mobile', 'tablet']:
        return device_type
    elif browser.startswith('Snapchat') and device_type == 'mobile':
        return device_type
    elif browser.startswith('Samsung Internet') and device_type in ['mobile', 'tablet']:
        return device_type
    elif browser.startswith('Safari') and device_type in ['mobile', 'tablet']:
        return device_type
    elif browser.startswith('Opera Mini') and device_type == 'mobile':
        return device_type
    elif browser.startswith('Opera Mobile') and device_type in ['mobile', 'tablet']:
        return device_type
    else:
        return 'Incompatible'




# Define batch size and initialize the LSTM model
batch_size = 50000
model = Sequential()
model.add(LSTM(64, input_shape=(1, 9)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

total_samples = 0
correct_predictions = 0
# Preprocess the dataset and train the model in batches
for chunk in pd.read_csv('C:/Users/anike/Downloads/Chitkara_Anomaly_Detection/Chitkara_Anomaly_Detection/Login_Data.csv', chunksize=batch_size):
    # Preprocess the chunk
    chunk['IPtoCountry'] = chunk['IP Address'].apply(lambda ip: get_geolocation(ip))
    chunk['Compatibility'] = chunk.apply(lambda row: check_compatibility(row['Browser Name and Version'], row['Device Type']), axis=1)
    chunk['Temp_label'] = np.where((chunk['IPtoCountry'] != chunk['Country']) | (chunk['Compatibility'] != chunk['Device Type']), 1, 0)
    chunk['Login_bin'] = chunk['Login Successful'].astype(int)
    chunk['Label_f'] = np.logical_not(chunk['Temp_label'] ^ chunk['Login_bin']).astype(int)

    chunk = chunk.drop(['Compatibility', 'IPtoCountry', 'Temp_label', 'Login_bin'], axis=1)
    chunk['Login Timestamp'] = pd.to_datetime(chunk['Login Timestamp'])
    chunk['Login Timestamp'] = chunk['Login Timestamp'].apply(lambda x: datetime.timestamp(x))
    chunk['IP Address'] = chunk['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))

    # print(chunk)

    categorical_columns = ['Country', 'Region', 'City', 'Browser Name and Version', 'Device Type']
    # Perform label encoding on each categorical column
    for column in categorical_columns:
          label_encoder = LabelEncoder()
          chunk[column] = label_encoder.fit_transform(chunk[column])


    temp = chunk
    temp = temp.drop(['Label_f'], axis=1)
    X = temp.values
    y = chunk['Label_f']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    # Train the model on the chunk
    model.fit(X, y, epochs=50, batch_size=25, verbose=0)
    # Evaluate the accuracy on the chunk
    predictions = model.predict(X)
    predicted_labels = np.round(predictions).flatten()
    correct_predictions += np.sum(predicted_labels == y)
    total_samples += len(y)

# Calculate accuracy
accuracy = correct_predictions / total_samples
print("Accuracy:", accuracy)



with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)