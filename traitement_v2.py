from flask import Flask, request
import numpy as np
from pickle import load

app = Flask(__name__)

# Charger le pipeline et le modèle
with open("scaler.pkl", "rb") as f:
    scaler = load(f)

with open("pca.pkl", "rb") as f:
    pca = load(f)

with open("bc_dep.pkl", "rb") as f:
    model = load(f)

# Ordre exact des features
feature_names = [
    'memory_problems', 'BMI', 'height', 'sleep_hours', 'weight',
    'platelet_count', 'cant_work', 'white_BCC', 'glucose', 'monocyte_percent',
    'iron', 'RBC_count', 'hematocrit', 'neutrophils_count', 'ALP',
    'trouble_sleeping_history', 'pulse', 'RDW', 'GGT', 'hemoglobin',
    'HDL', 'sedentary_time', 'household_income', 'BUN', 'albumin',
    'bilirubin', 'prescriptions_count', 'drinks_per_occasion', 'drinks_past_year',
    'start_smoking_age', 'time_in_current_job', 'education_level',
    'current_cigarettes_per_day', 'marital_status', 'arthritis_onset'
]

@app.route('/')
def home():
    return open('formulaire_depression.html').read()

@app.route('/resultats.php', methods=['POST'])
def predict():

    # Récupérer les valeurs du formulaire dans le bon ordre
    user_input = []
    for feature in feature_names:
        value = request.form.get(feature, 0)
        if value == "":
            value = 0
        user_input.append(float(value))

    # Passer dans le pipeline: scaler -> pca -> modèle
    X = np.array(user_input).reshape(1, -1)
    X = scaler.transform(X)
    X = pca.transform(X)

    # Prédiction: 0 = Depressed, 1 = Not Depressed
    prediction = model.predict(X)[0]

    if prediction == 0:
        result = "Depressed"
        color = "red"
    else:
        result = "Not Depressed"
        color = "green"

    return f"""
    <html>
    <head><title>Results</title></head>
    <body>
        <h1>Your Results</h1>
        <fieldset>
            <legend>Prediction</legend>
            <p>Based on your answers, our model estimates that you are:</p>
            <h2 style="color: {color};">{result}</h2>
            <p style="color: red;">
                <em>This result does not replace the advice of a healthcare professional.
                If you think you may be suffering from depression, please consult a doctor.</em>
            </p>
        </fieldset>
        <br/>
        <a href="/">Go back to the form</a>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)