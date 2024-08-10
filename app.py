from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model                                                                                                                                                                                                                                                          #type:ignore
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)

model = load_model('ann_model.h5')

client = MongoClient('mongodb+srv://Textovert:Kyundu17@textovert.uzlevw3.mongodb.net/')
db = client['loan_application']
collection = db['applications']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('loan-application.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        no_of_dependents = int(request.form['no_of_dependents'])
        income_annum = float(request.form['income_annum'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        cibil_score = int(request.form['cibil_score'])

        residential_assets_value = float(request.form['residential_assets_value'])
        commercial_assets_value = float(request.form['commercial_assets_value'])
        luxury_assets_value = float(request.form['luxury_assets_value'])
        bank_asset_value = float(request.form['bank_asset_value'])

        education_not_graduate = 1 if request.form['education_NotGraduate'] == 'Yes' else 0
        self_employed_yes = 1 if request.form['self_employed_Yes'] == 'Yes' else 0

        user_input = {
            'no_of_dependents': no_of_dependents,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value,
            'education_not_graduate': education_not_graduate,
            'self_employed_yes': self_employed_yes
        }
        result = collection.insert_one(user_input)
        user_id = result.inserted_id

        fetched_data = collection.find_one(sort=[('_id', -1)])  
        if not fetched_data:
            return "No data found in the database."

        features = np.array([
            fetched_data['no_of_dependents'],
            fetched_data['income_annum'],
            fetched_data['loan_amount'],
            fetched_data['loan_term'],
            fetched_data['cibil_score'],
            fetched_data['residential_assets_value'],
            fetched_data['commercial_assets_value'],
            fetched_data['luxury_assets_value'],
            fetched_data['bank_asset_value'],
            fetched_data['education_not_graduate'],
            fetched_data['self_employed_yes']
        ]).reshape(1, -1)

        prediction = model.predict(features)
        prediction = (prediction > 0.5).astype(int)[0][0]

        prediction_text = "Approved" if prediction == 1 else "Rejected"

        return prediction_text

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
