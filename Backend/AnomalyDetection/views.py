import ast
from rest_framework.decorators import api_view
from django.http import JsonResponse
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import ipaddress
from sklearn.preprocessing import LabelEncoder, StandardScaler


@api_view(['POST'])
def check_anomaly(record):
    # If receiving data from browser
    # datalist=record.data

    # If receiving data from postman
    dataline = record.data.get('data')
    datalist = ast.literal_eval(dataline)
    # Preprocess the input record
    attributes = ['Login Timestamp', 'User ID', 'IP Address', 'Country', 'Region', 'City', 'Browser Name and Version',
                  'Device Type', 'Login Successful']
    expected_columns = [
        'loginTimestamp', 'userID', 'ipAddress', 'country', 'region', 'city',
        'browserName', 'deviceType', 'loginSuccessful'
    ]

    feature_mapping = {
        'loginTimestamp': 'Login Timestamp',
        'userID': 'User ID',
        'ipAddress': 'IP Address',
        'country': 'Country',
        'region': 'Region',
        'city': 'City',
        'browserName': 'Browser Name and Version',
        'deviceType': 'Device Type',
        'loginSuccessful': 'Login Successful'
    }

    # If the data contains column names, use them to create the DataFrame
    if isinstance(datalist, dict) and set(expected_columns).issubset(datalist.keys()):
        mapped_datalist = {feature_mapping.get(key, key): value for key, value in datalist.items()}
    else:
        # If the data doesn't contain column names, assume it's a list and create the DataFrame
        mapped_datalist = {attributes[i]: datalist[i] for i in range(len(attributes))}

    df = pd.DataFrame([mapped_datalist], columns=attributes)
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], format='%d/%m/%Y %H:%M:%S')
    df['Login Timestamp'] = df['Login Timestamp'].apply(lambda x: datetime.timestamp(x))
    df['IP Address'] = df['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))
    categorical_columns = ['Country', 'Region', 'City', 'Browser Name and Version', 'Device Type']

    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

    Z = df.values

    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)
    Z = np.reshape(Z, (Z.shape[0], 1, Z.shape[1]))

    with open('./AnomalyDetection/model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Predict the anomaly using the trained model
    prediction = model.predict(Z)
    predicted_value = np.round(prediction).flatten().astype(int)

    if predicted_value == 0:
        return JsonResponse({"result": "Not an Anomaly"})
    else:
        return JsonResponse({"result": "Anomaly"})
