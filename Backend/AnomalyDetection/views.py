import ast
import ipaddress
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from rest_framework.decorators import api_view
from django.http import JsonResponse


@api_view(['POST'])
def check_anomaly(request):
    # If receiving data from browser
    # datalist=request.data

    # If receiving data from postman
    dataline = request.data.get('data')
    datalist = ast.literal_eval(dataline)

    columns = [
        'Login Timestamp', 'User ID', 'IP Address', 'Country', 'Region', 'City',
        'Browser Name and Version', 'Device Type', 'Login Successful'
    ]

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
        # dataonerow = pd.DataFrame([mapped_datalist])
    else:
        # If the data doesn't contain column names, assume it's a list and create the DataFrame
        # dataonerow = pd.DataFrame([datalist], columns=columns)
        mapped_datalist = {columns[i]: datalist[i] for i in range(len(columns))}

    # dataonerow = pd.DataFrame([datalist], columns=columns)
    dataonerow = pd.DataFrame([mapped_datalist], columns=columns)

    dataonerow['Login Timestamp'] = pd.to_datetime(dataonerow['Login Timestamp'], format='%m-%d-%Y %I.%M.%S %p')
    dataonerow['Login Timestamp'] = dataonerow['Login Timestamp'].apply(lambda x: datetime.timestamp(x))
    dataonerow['IP Address'] = dataonerow['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))

    features1 = ['Login Timestamp', 'User ID', 'IP Address', 'Login Successful']
    categorical_features1 = ['Country', 'Region', 'City', 'Browser Name and Version', 'Device Type']

    dataonerow = pd.get_dummies(dataonerow, columns=categorical_features1)

    samtest = dataonerow[features1]
    anomaly_score = 0
    with open('./AnomalyDetection/main.pkl', 'rb') as file:
        model = pickle.load(file)
    sampredict = model.predict(samtest)
    anomaly_score = model.decision_function(samtest)
    print(f"Anomaly score: {anomaly_score[0]}")

    if sampredict[0] == 1:
        return JsonResponse({"result": "Not an Anomaly"})
    else:
        print("Anomaly detected. The features responsible for the anomaly are:")
        features_imp = np.array(samtest.columns)[anomaly_score[0] < 0]
        print(features_imp)
        return JsonResponse({"result": "Anomaly"})
