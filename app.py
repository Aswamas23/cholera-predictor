from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('cholera_model.pkl')
le_region = joblib.load('region_encoder.pkl')
le_target = joblib.load('target_encoder.pkl')

# The order of features as used in training
feature_cols = [
    'Region_encoded', 'Month', 'Avg_AirTemp_C', 'Rainfall_mm',
    'Water_Quality_Index', 'Sanitation_Index', 'Population_Density'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            region = request.form['region']
            month = int(request.form['month'])
            avg_temp = float(request.form['avg_temp'])
            rainfall = float(request.form['rainfall'])
            water_quality = int(request.form['water_quality'])
            sanitation = int(request.form['sanitation'])
            pop_density = int(request.form['pop_density'])

            # Encode region as done during training
            region_encoded = le_region.transform([region])[0]

            # Prepare input DataFrame
            input_df = pd.DataFrame([[
                region_encoded, month, avg_temp, rainfall,
                water_quality, sanitation, pop_density
            ]], columns=feature_cols)

            # Predict
            pred_encoded = model.predict(input_df)[0]
            prediction = le_target.inverse_transform([pred_encoded])[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
