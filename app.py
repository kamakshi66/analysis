import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.io as pio
import json

# Initialize Flask app
app = Flask(__name__)

# Load the preprocessed data and trained model
with open('olympics_data.pkl', 'rb') as file:
    df = pickle.load(file)

with open('olympics_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions
def predict_medal(age, height, weight, year):
    input_data = pd.DataFrame([[age, height, weight, year]], columns=['Age', 'Height', 'Weight', 'Year'])
    prediction = model.predict(input_data)
    medal = {3: 'Gold', 2: 'Silver', 1: 'Bronze'}
    return medal[prediction[0]]

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for medal prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        year = int(request.form['year'])

        # Predict the medal
        medal = predict_medal(age, height, weight, year)
        return render_template('index.html', prediction_text=f"Predicted Medal: {medal}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Route for model evaluation
@app.route('/evaluate')
def evaluate():
    # Prepare data for evaluation
    X = df[['Age', 'Height', 'Weight', 'Year']].fillna(df[['Age', 'Height', 'Weight', 'Year']].mean())
    y = df['Medal']
    
    # Evaluation metrics
    y_pred = model.predict(X)
    accuracy = accuracy_sc
