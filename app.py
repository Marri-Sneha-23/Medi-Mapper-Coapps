from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessing pipeline
dtr = joblib.load('decision_tree_model.pkl')
preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')

# Define diseases list
diseases = [
    "Acne", "Osteoarthritis", "Bronchial Asthma", "Alcoholic hepatitis", "Impetigo",
    "Tonsillitis", "(Vertigo) Paroxysmal Positional Vertigo", "Dimorphic hemorrhoids (piles)",
    "Tuberculosis", "Pneumonia", "Varicose veins", "Hypothyroidism", "Heart attack",
    "Hypoglycemia", "Cervical spondylosis", "Diabetes", "Common Cold", "Arthritis",
    "Hypertension", "Chronic cholestasis", "Migraine", "Urinary tract infection",
    "Hyperthyroidism", "GERD (Gastroesophageal Reflux Disease)", "Allergy", "Chickenpox",
    "Dengue", "Psoriasis", "Malaria", "Fungal infection", "Jaundice", "Hepatitis A",
    "Paralysis (brain hemorrhage)", "Peptic ulcer disease", "Vertigo (Paroxysmal Positional Vertigo)",
    "Hepatitis B", "Gastroenteritis", "Typhoid", "AIDS", "Hepatitis E", "Drug Reaction",
    "Hepatitis C", "Hepatitis D"
]

@app.route('/')
def index():
    return render_template('index.html', diseases=diseases)

@app.route('/recommend', methods=['POST'])
def recommend():
    age = int(request.form['age'])
    disease = request.form['disease']
    test_result = request.form['test_result']

    input_data = pd.DataFrame({
        'Age': [age],
        'Disease': [disease],
        'Test Result': [test_result]
    })

    preprocessed_input = preprocessing_pipeline.transform(input_data)
    medication = dtr.predict(preprocessed_input)

    return render_template('recommendation.html', medication=medication[0])

if __name__ == '__main__':
    app.run(debug=True)
