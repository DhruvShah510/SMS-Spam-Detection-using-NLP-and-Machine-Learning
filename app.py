from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Load the trained pipeline
pipeline = joblib.load('spam_detection_pipeline.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        message = request.form["message"]
        # Convert to pandas Series so .apply() works
        pred = pipeline.predict(pd.Series([message]))[0]
        prediction = "Spam" if pred == 1 else "Not Spam"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
