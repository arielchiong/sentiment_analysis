from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import threading
import os
import time

# Initialize the Flask app, specifying the custom template folder
app = Flask(__name__, template_folder='./web')

model = None
tokenizer = None
model_ready = False

def load_model():
    global model, tokenizer, model_ready
    model = tf.keras.models.load_model('models/sentiment_model.keras')
    tokenizer_path = 'models/tokenizer.json'
    with open(tokenizer_path, 'r') as f:
        data = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    model_ready = True

# Run the model loading in a separate thread
thread = threading.Thread(target=load_model)
thread.start()

# Function to preprocess input text and predict sentiment
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive review' if prediction >= 0.5 else 'Negative review'
    return sentiment

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if not model_ready:
        return render_template('loading.html')
    
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('index.html', review=review, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)