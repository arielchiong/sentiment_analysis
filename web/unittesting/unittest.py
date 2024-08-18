import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained sentiment analysis model
model = tf.keras.models.load_model('models/sentiment_model.keras')

# Load the tokenizer from the JSON file
with open('models/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

# Function to preprocess the input text
def preprocess_text(text):
    # Convert the text to a sequence of integers using the tokenizer
    sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence to ensure it has the correct length (200, as used in training)
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    return padded_sequence

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Make a prediction using the loaded model
    prediction = model.predict(processed_text)
    # Interpret the prediction (assume a threshold of 0.5 for binary classification)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return sentiment

# Example usage: test the model with some input texts
example_texts = [
    "I absolutely love this product! It works perfectly.",
    "This is the worst thing I've ever bought.",
    "Not bad, but could be better.",
    "The quality is amazing for the price.",
    "I wouldn't recommend this to anyone."
]

for text in example_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")