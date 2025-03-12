from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='../templates')

model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    input_data = None

    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

    return render_template('index.html', 
                         prediction=prediction, 
                         probability=probability, 
                         input_data=input_data.to_dict('records')[0] if input_data is not None else None)