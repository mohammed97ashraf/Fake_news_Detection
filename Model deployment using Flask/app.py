from flask import Flask, render_template, request
import joblib
import re
import string
import pandas as pd
import os

app = Flask(__name__)

# Correct the path to the model file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(parent_dir, "Model.pkl")
print(f"Corrected model path: {model_path}")

# Load the model
Model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template("index.html")

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/', methods=['POST'])
def pre():
    if request.method == 'POST':
        txt = request.form['txt']
        txt = wordpre(txt)
        txt = pd.Series([txt])  # Ensure this is passed as a list or a Series
        
        try:
            result = Model.predict(txt)
            result = result[0]  # Assuming it's a list/array, get the first prediction
        except Exception as e:
            result = f"Error: {str(e)}"
        
        return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
