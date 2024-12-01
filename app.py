# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import pickle
# import numpy as np
# import os

# # Load the trained model
# model_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/random_forest_model (1).pkl'  # Update this to the correct path
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# # Load the dataset (for calculating derived features)
# dataset_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/df_fff (2).csv'  # Update this to the correct path
# data = pd.read_csv(dataset_path)

# # Initialize Flask app
# app = Flask(__name__)

# # Calculate additional features
# def calculate_features(input_features):
#     tempmax = input_features['tempmax']
#     tempmin = input_features['tempmin']
#     feelslikemax = input_features['feelslikemax']
#     feelslikemin = input_features['feelslikemin']
    
#     # Derived features
#     temp_range = tempmax - tempmin
#     heat_index = (feelslikemax + feelslikemin) / 2
    
#     # Add rolling demand features (assuming dataset includes POWER_DEMAND)
#     data['POWER_DEMAND_rolling_3day'] = data['POWER_DEMAND'].rolling(window=3).mean()
#     data['POWER_DEMAND_rolling_7day'] = data['POWER_DEMAND'].rolling(window=7).mean()
    
#     # Use the last values for rolling features (simulate recent state)
#     rolling_3day = data['POWER_DEMAND_rolling_3day'].iloc[-1]
#     rolling_7day = data['POWER_DEMAND_rolling_7day'].iloc[-1]
    
#     # Return the full feature set
#     return {
#         'tempmax': tempmax,
#         'tempmin': tempmin,
#         'feelslikemax': feelslikemax,
#         'feelslikemin': feelslikemin,
#         'humidity': input_features['humidity'],
#         'windspeed': input_features['windspeed'],
#         'heat_index': heat_index,
#         'POWER_DEMAND_rolling_3day': rolling_3day,
#         'POWER_DEMAND_rolling_7day': rolling_7day,
#         'temp_range': temp_range
#     }

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')  # Create index.html for input form

# # Manual input prediction route
# @app.route('/predict_manual', methods=['POST'])
# def predict_manual():
#     # Extract input data from the form
#     input_features = {
#         'tempmax': float(request.form['tempmax']),
#         'tempmin': float(request.form['tempmin']),
#         'feelslikemax': float(request.form['feelslikemax']),
#         'feelslikemin': float(request.form['feelslikemin']),
#         'humidity': float(request.form['humidity']),
#         'windspeed': float(request.form['windspeed']),
#     }
    
#     # Calculate derived features
#     features = calculate_features(input_features)
    
#     # Create input DataFrame for the model
#     input_df = pd.DataFrame([features])
    
#     # Predict power demand
#     prediction = model.predict(input_df)[0]
    
#     # Return the prediction result
#     return render_template('result.html', prediction=prediction)

# # Prediction route for uploading CSV
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if file and file.filename.endswith('.csv'):
#         # Read the uploaded CSV file into a DataFrame
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
#         df = pd.read_csv(file_path)

#         # Check if required columns are present
#         required_columns = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'humidity', 'windspeed']
#         if not all(col in df.columns for col in required_columns):
#             return jsonify({'error': f'CSV file must contain the following columns: {", ".join(required_columns)}'})

#         # Calculate derived features for the entire DataFrame
#         features_list = []
#         for index, row in df.iterrows():
#             input_features = {
#                 'tempmax': row['tempmax'],
#                 'tempmin': row['tempmin'],
#                 'feelslikemax': row['feelslikemax'],
#                 'feelslikemin': row['feelslikemin'],
#                 'humidity': row['humidity'],
#                 'windspeed': row['windspeed'],
#             }
#             features = calculate_features(input_features)
#             features_list.append(features)

#         # Create a DataFrame with the calculated features
#         features_df = pd.DataFrame(features_list)

#         # Make predictions
#         predictions = model.predict(features_df)

#         # Add the predictions to the DataFrame
#         df['Predicted_Power_Demand'] = predictions

#         # Return the DataFrame with the predictions
#         result = df.to_dict(orient='records')
#         return jsonify(result)

#     return jsonify({'error': 'Invalid file format. Please upload a CSV file.'})

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# ver 2
# from flask import Flask, request, render_template, jsonify, current_app
# import pandas as pd
# import pickle
# import os
# import secrets
# from PIL import Image
# from sklearn.impute import SimpleImputer

# def calculate_heat_index(temp, humidity):
#     """Calculates the heat index based on temperature and humidity."""
#     c1 = -42.379
#     c2 = 2.04901523
#     c3 = 10.14333127
#     c4 = -0.22475541
#     c5 = -6.83783 * 10**-3
#     c6 = -5.481717 * 10**-2
#     c7 = 1.22874 * 10**-3
#     c8 = 8.5282 * 10**-4
#     c9 = -1.99 * 10**-6

#     T = temp  # Temperature in Celsius
#     RH = humidity  # Relative humidity in percentage

#     heat_index = c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + c6*RH**2 + c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2
#     return heat_index

# # Load the trained model
# model_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/random_forest_model (1).pkl'  # Update this to the correct path
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# # Load the dataset (for calculating derived features)
# dataset_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/df_fff (2).csv'  # Update this to the correct path
# data = pd.read_csv(dataset_path)

# # Initialize Flask app
# app = Flask(__name__)

# # File upload configuration
# UPLOAD_FOLDER = 'uploads'  # Directory for storing uploaded files (CSV or images)
# ALLOWED_EXTENSIONS = {'csv'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Calculate additional features
# def calculate_features(input_features):
#     tempmax = input_features['tempmax']
#     tempmin = input_features['tempmin']
#     feelslikemax = input_features['feelslikemax']
#     feelslikemin = input_features['feelslikemin']
    
#     # Derived features
#     temp_range = tempmax - tempmin
#     temp = input_features['temp']
#     humidity = input_features['humidity']
#     heat_index = calculate_heat_index(temp , humidity)
    
#     # Add rolling demand features (assuming dataset includes POWER_DEMAND)
#     data['POWER_DEMAND_rolling_3day'] = data['POWER_DEMAND'].rolling(window=3).mean().shift(1)
#     data['POWER_DEMAND_rolling_7day'] = data['POWER_DEMAND'].rolling(window=7).mean().shift(1)
    
#     # Use the last values for rolling features (simulate recent state)
#     rolling_3day = data['POWER_DEMAND_rolling_3day'].iloc[-1]
#     rolling_7day = data['POWER_DEMAND_rolling_7day'].iloc[-1]
    
#     # Return the full feature set
#     return {
#         'tempmax': [float(tempmax)],
#         'tempmin': [float(tempmin)],
#         'feelslikemax': [float(feelslikemax)],
#         'feelslikemin': [float(feelslikemin)],
#         'humidity': [float(input_features['humidity'])],
#         'windspeed': [float(input_features['windspeed'])],
#         'heat_index': [float(heat_index)],
#         'POWER_DEMAND_rolling_3day': [float(rolling_3day)],
#         'POWER_DEMAND_rolling_7day': [float(rolling_7day)],
#         'temp_range': [float(temp_range)],
        
#     }

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')  # Create index.html for input form

# # Manual input prediction route
# @app.route('/predict_manual', methods=['POST'])
# def predict_manual():
#     # Extract input data from the form
#     imputer = SimpleImputer(strategy='mean')
#     input_features = {
#         'tempmax': float(request.form['tempmax']),
#         'tempmin': float(request.form['tempmin']),
#         'feelslikemax': float(request.form['feelslikemax']),
#         'feelslikemin': float(request.form['feelslikemin']),
#         'humidity': float(request.form['humidity']),
#         'windspeed': float(request.form['windspeed']),
#     }
    
#     # Calculate derived features
#     features = calculate_features(input_features)
    
#     # Create input DataFrame for the model
#     input_df = pd.DataFrame([features])
#     input_df_imputer = imputer.fit_transform(input_df)
#     # Predict power demand
#     prediction = model.predict(input_df_imputer)[0]
    
#     # Return the prediction result
#     return render_template('result.html', prediction=prediction)

# # Function to save uploaded files (CSV or image)
# def save_file(file, folder, allowed_extensions):
#     """
#     Save the uploaded file to the server and return the filename.

#     :param file: The uploaded file (CSV or image)
#     :param folder: Folder where the file should be saved
#     :param allowed_extensions: Set of allowed file extensions
#     :return: The filename of the saved file
#     """
#     if file and allowed_file(file.filename, allowed_extensions):
#         # Secure the filename and ensure the directory exists
#         filename = secrets.token_hex(8) + os.path.splitext(file.filename)[1]
#         file_path = os.path.join(folder, filename)
#         file.save(file_path)
#         return filename
#     return None

# # Function to check allowed file extensions
# def allowed_file(filename, allowed_extensions):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# # Prediction route for uploading CSV
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Save the uploaded file
#     filename = save_file(file, app.config['UPLOAD_FOLDER'], ALLOWED_EXTENSIONS)
#     if filename is None:
#         return jsonify({'error': 'Invalid file format. Please upload a CSV file.'})

#     # Read the uploaded CSV file into a DataFrame
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     df = pd.read_csv(file_path)

#     # Check if required columns are present
#     required_columns = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'humidity', 'windspeed']
#     if not all(col in df.columns for col in required_columns):
#         return jsonify({'error': f'CSV file must contain the following columns: {", ".join(required_columns)}'})

#     # Calculate derived features for the entire DataFrame
#     features_list = []
#     for index, row in df.iterrows():
#         input_features = {
#             'tempmax': row['tempmax'],
#             'tempmin': row['tempmin'],
#             'feelslikemax': row['feelslikemax'],
#             'feelslikemin': row['feelslikemin'],
#             'humidity': row['humidity'],
#             'windspeed': row['windspeed'],
#         }
#         features = calculate_features(input_features)
#         features_list.append(features)

#     # Create a DataFrame with the calculated features
#     features_df = pd.DataFrame(features_list)

#     # Make predictions
#     predictions = model.predict(features_df)

#     # Add the predictions to the DataFrame
#     df['Predicted_Power_Demand'] = predictions

#     # Return the DataFrame with the predictions
#     result = df.to_dict(orient='records')
#     return jsonify(result)

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# ver 3

from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import os
import secrets
from PIL import Image
from sklearn.impute import SimpleImputer

def calculate_heat_index(temp, humidity):
    """Calculates the heat index based on temperature and humidity."""
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 * 10**-3
    c6 = -5.481717 * 10**-2
    c7 = 1.22874 * 10**-3
    c8 = 8.5282 * 10**-4
    c9 = -1.99 * 10**-6

    T = temp  # Temperature in Celsius
    RH = humidity  # Relative humidity in percentage

    heat_index = c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + c6*RH**2 + c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2
    return heat_index

# Load the trained model
model_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/random_forest_model (1).pkl'  # Update this to the correct path
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset (for calculating derived features)
dataset_path = 'DELHI-Peek-ELECTRIC-Demand-prediction/df_fff (2).csv'  # Update this to the correct path
data = pd.read_csv(dataset_path)

# Initialize Flask app
app = Flask(__name__)

# File upload configuration
UPLOAD_FOLDER = 'uploads'  # Directory for storing uploaded files (CSV or images)
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Calculate additional features
def calculate_features(input_features):
    tempmax = input_features['tempmax']
    tempmin = input_features['tempmin']
    feelslikemax = input_features['feelslikemax']
    feelslikemin = input_features['feelslikemin']
    temp = input_features['temp']
    humidity = input_features['humidity']

    # Derived features
    temp_range = tempmax - tempmin
    heat_index = calculate_heat_index(temp, humidity)

    # Add rolling demand features (assuming dataset includes POWER_DEMAND)
    data['POWER_DEMAND_rolling_3day'] = data['POWER_DEMAND'].rolling(window=3).mean().shift(1)
    data['POWER_DEMAND_rolling_7day'] = data['POWER_DEMAND'].rolling(window=7).mean().shift(1)

    # Use the last values for rolling features (simulate recent state)
    rolling_3day = data['POWER_DEMAND_rolling_3day'].iloc[-1]
    rolling_7day = data['POWER_DEMAND_rolling_7day'].iloc[-1]

    # Return the full feature set
    return {
        'tempmax': [float(tempmax)],
        'tempmin': [float(tempmin)],
        'feelslikemax': [float(feelslikemax)],
        'feelslikemin': [float(feelslikemin)],
        'humidity': [float(humidity)],
        'windspeed': [float(input_features['windspeed'])],
        'heat_index': [float(heat_index)],
        'POWER_DEMAND_rolling_3day': [float(rolling_3day)],
        'POWER_DEMAND_rolling_7day': [float(rolling_7day)],
        'temp_range': [float(temp_range)],
       
    }

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Create index.html for input form

# Manual input prediction route
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    imputer = SimpleImputer(strategy='mean')
    # Extract input data from the form
    input_features = {
        'tempmax': float(request.form['tempmax']),
        'tempmin': float(request.form['tempmin']),
        'feelslikemax': float(request.form['feelslikemax']),
        'feelslikemin': float(request.form['feelslikemin']),
        'humidity': float(request.form['humidity']),
        'windspeed': float(request.form['windspeed']),
        'temp': float(request.form['temp'])  # Added 'temp'
    }

    # Calculate derived features
    features = calculate_features(input_features)

    # Create input DataFrame for the model
    input_df = pd.DataFrame(features)
    input_df_imputed = imputer.fit_transform(input_df)

    # Predict power demand
    prediction = model.predict(input_df_imputed)[0]

    # Return the prediction result
    return render_template('result.html', prediction=prediction)

# Function to save uploaded files (CSV or image)
def save_file(file, folder, allowed_extensions):
    """
    Save the uploaded file to the server and return the filename.

    :param file: The uploaded file (CSV or image)
    :param folder: Folder where the file should be saved
    :param allowed_extensions: Set of allowed file extensions
    :return: The filename of the saved file
    """
    if file and allowed_file(file.filename, allowed_extensions):
        filename = secrets.token_hex(8) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(folder, filename)
        file.save(file_path)
        return filename
    return None

# Function to check allowed file extensions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Prediction route for uploading CSV
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    filename = save_file(file, app.config['UPLOAD_FOLDER'], ALLOWED_EXTENSIONS)
    if filename is None:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'})

    # Read the uploaded CSV file into a DataFrame
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    # Check if required columns are present
    required_columns = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'humidity', 'windspeed', 'temp']
    if not all(col in df.columns for col in required_columns):
        return jsonify({'error': f'CSV file must contain the following columns: {", ".join(required_columns)}'})

    # Impute missing values in the DataFrame (if any)
    imputer = SimpleImputer(strategy='mean')
    df[required_columns] = imputer.fit_transform(df[required_columns])

    # Calculate derived features for the entire DataFrame
    features_list = []
    for index, row in df.iterrows():
        input_features = {
            'tempmax': row['tempmax'],
            'tempmin': row['tempmin'],
            'feelslikemax': row['feelslikemax'],
            'feelslikemin': row['feelslikemin'],
            'humidity': row['humidity'],
            'windspeed': row['windspeed'],
            'temp': row['temp']
        }
        # Calculate features for each row

        features = calculate_features(input_features)
        features.pop('temp', None)
        features_list.append(features)

    # Flatten the list of dictionaries into a DataFrame
    features_df = pd.concat([pd.DataFrame(feature) for feature in features_list], ignore_index=True)

    # Ensure alignment of columns for prediction
    # features_df = features_df.reindex(columns=model.feature_names_in, fill_value=0)
    # features_df['temp'].drop(axis = 0 , inplace=True)
    # Make predictions
    predictions = model.predict(features_df)

    # Add the predictions to the original DataFrame
    df['Predicted_Power_Demand'] = predictions

    # Return the DataFrame with the predictions
    result = df.to_dict(orient='records')
    # return jsonify(result)
    return render_template('recommendation.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
