from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.form
        features = [
            float(data['age']),
            float(data['gender']),
            float(data['sleep_duration']),
            float(data['dietary_habits']),
            float(data['suicidal_thoughts']),
            float(data['family_history']),
            float(data['cgpa']),
            float(data['academic_pressure']),
            float(data['work_pressure']),
            float(data['work_study_hours']),
            float(data['financial_stress']),
            float(data['study_satisfaction']),
            float(data['job_satisfaction']),
        ]
        features_scaled = scaler.transform([features])
        prediction = int(model.predict(features_scaled)[0])
        result = "High Risk of Depression 😟" if prediction == 0 else "Low Risk of Depression 🙂"

        return jsonify({'prediction': result})

    return render_template("index.html")

@app.route("/booking")
def booking():
    return render_template("booking.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        data = request.form
        features = [
            float(data['age']),
            float(data['gender']),
            float(data['sleep_duration']),
            float(data['dietary_habits']),
            float(data['suicidal_thoughts']),
            float(data['family_history']),
            float(data['cgpa']),
            float(data['academic_pressure']),
            float(data['work_pressure']),
            float(data['work_study_hours']),
            float(data['financial_stress']),
            float(data['study_satisfaction']),
            float(data['job_satisfaction']),
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        result = "High Risk of Depression 😟" if prediction == 1 else "Low Risk of Depression 🙂"
        return jsonify({'prediction': result})
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)
