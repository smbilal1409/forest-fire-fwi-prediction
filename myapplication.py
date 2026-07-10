
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

myapplication = Flask(__name__)
app = myapplication

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scalar_model = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('Home.html')


@app.route("/dataprediction", methods=["GET", "POST"])
def dataprediction():

    if request.method == "POST":

        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])
        Region = float(request.form["Region"])

        data = np.array([[Temperature, RH, Ws, Rain,
                          FFMC, DMC, ISI, Classes, Region]])

        scaled = standard_scalar_model.transform(data)

        prediction = ridge_model.predict(scaled)[0]

        if prediction < 5:
            risk_label = "🟢 LOW"
            risk_class = "low"
        elif prediction < 15:
            risk_label = "🟡 MODERATE"
            risk_class = "moderate"
        elif prediction < 30:
            risk_label = "🟠 HIGH"
            risk_class = "high"
        else:
            risk_label = "🔴 EXTREME"
            risk_class = "extreme"

        return render_template(
            "Home.html",
            prediction=round(prediction, 2),
            risk=risk_label,
            risk_class=risk_class
        )

    return render_template("Home.html")


# Define an API route returning JSON data
@app.route("/api/status")
def get_status():
    return jsonify({"status": "active", "framework": "Flask"})


if __name__ == "__main__":
    app.run(host="0.0.0.0")