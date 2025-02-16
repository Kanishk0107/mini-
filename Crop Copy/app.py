from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    features = [
        float(request.form["nitrogen"]),
        float(request.form["phosphorous"]),
        float(request.form["potassium"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    # Convert to NumPy array and reshape for model
    input_data = np.array([features]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Render result.html with prediction
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
