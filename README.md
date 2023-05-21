# Byte_Busters

STGI Hackathon

# Anomaly Detection using Machine Learning

This project is developed for a hackathon and focuses on anomaly detection using machine learning techniques. The project includes a user interface (UI) implemented in HTML and JavaScript, which allows users to input login information and sends it to a backend server for anomaly detection.

## Getting Started

To run the project locally, follow the instructions below:

### Prerequisites

    -Python (version 3.x)
    -Django (version 4.x)

## Installation

1. Clone this repository

   ```bash
   git clone https://github.com/Pranav-github02/Byte_Busters.git

   ```

2. Change to the project directory:

   ```bash
   cd anomaly-detection

   ```

3. Install the required Python packages
4. Start the Django server:

   ```bash
   python manage.py runserver

   -The server will start running at http://127.0.0.1:8000/.
   ```

## Usage

1. Open a web browser and navigate to http://127.0.0.1:5500/index.html.

2. The UI will be displayed, allowing you to input login information.

3. Fill in the required fields, including the login timestamp, user ID, IP address, country, region, city, browser name and version, device type, and whether the login was successful.
   sample data -> ['02-03-2020 12.46.00 PM', -9.05E+18, '139.164.54.1', 'NO', '-', '-', 'Chrome 79.0.3945.192.203', 'desktop', True]

4. Click the "Submit" button.

5. The UI will send the form data to the backend server for anomaly detection.

6. The server will process the data using the machine learning model and return the results.

7. The results will be displayed in the result area of the UI, showing the detected anomalies.

## Backend Implementation

The backend server is implemented using Django, a Python web framework. When the form is submitted, the JavaScript code sends a POST request to the backend server's /anomaly/ endpoint with the form data in JSON format.

The backend server receives the data and applies the anomaly detection machine learning model to identify any anomalies. The specific implementation details of the machine learning model are not included in this project, and you should replace the placeholder code in the backend with your own anomaly detection algorithm.

Once the anomaly detection is performed, the server sends the response back to the UI, which then displays the results in the result area.

## Acknowledgments

This project was developed for the STGI Hackathon.
Special thanks to the organizers and mentors for their support and guidance.
