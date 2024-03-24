from flask import Flask,render_template,request
import joblib
import re
import os
import numpy as np
from textblob import TextBlob

app = Flask(__name__)

models_folder = "model"

# Specify the name of the model you want to load (without the file extension)
model_name = 'Logistic Regression'

# Construct the path to the saved model
saved_model_path = os.path.join(os.getcwd(), models_folder, f"{model_name}_model.pkl")

# Load the saved model
model = joblib.load(saved_model_path)

# Define preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove non-word characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from the start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with single space
    text = text.lower()  # Convert to lowercase
    return text

@app.route("/",methods=['GET','POST'])
def index(): 
        return render_template("home.html")
        
@app.route("/prediction",methods=['POST'])
def results():
        if request.method == 'POST':
            review = request.form['review']
            data_point = preprocess_text(review)
            prediction = model.predict([data_point])
            sentiment = "Positive Review 😁 Customer is staified.." if prediction[0] == 'Positive' else "Negative Review ☹️ Customer is dissatisfied.."
            return render_template('output.html',Sentiment = sentiment,review=review)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)