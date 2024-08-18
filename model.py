import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import numpy as np  # Importing numpy for numerical operations.

# Base directory for review files
base_directory = 'data/'

# Load the data
def load_reviews(base_dir):
    labeled_reviews = []
    unlabeled_reviews = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.review'):
                file_path = os.path.join(root, file)
                label = 'positive' if 'positive' in file_path else 'negative' if 'negative' in file_path else None
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        review = line.strip()
                        if label:
                            labeled_reviews.append((review, label))
                        else:
                            unlabeled_reviews.append(review)
    return labeled_reviews, unlabeled_reviews

# Load all reviews
labeled_reviews, unlabeled_reviews = load_reviews(base_directory)

# Convert labeled reviews to DataFrame
df = pd.DataFrame(labeled_reviews, columns=['review', 'label'])

# Display the first few rows of the DataFrame to check the data
print(df.head())

# Save the unlabeled reviews for potential future use
unlabeled_df = pd.DataFrame(unlabeled_reviews, columns=['review'])
unlabeled_df.to_csv('data/unlabeled_reviews.csv', index=False)

# Enhanced data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning to the reviews
df['review'] = df['review'].apply(clean_text)

# Tokenize and encode text
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

# Encode labels
df['label'] = df['label'].map({'positive': 1, 'negative': 0})

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model with pre-trained embeddings (optional, if available)
embedding_dim = 50
# Create an embedding matrix (optional step if you have pre-trained embeddings like GloVe or Word2Vec)
# embedding_matrix = ...

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=200),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),  # Reduced LSTM units
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # Reduced LSTM units
    tf.keras.layers.Dense(32, activation='relu'),  # Reduced Dense layer units
    tf.keras.layers.Dropout(0.7),  # Increased dropout rate
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a higher learning rate
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model with fewer epochs

# Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define the batch size
batch_size = 32

# Train the model for more epochs
history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
print(f'Test Loss (Binary Crossentropy): {test_loss}')

# Make predictions on the test set
y_pred_probs = model.predict(X_test)  # Get predicted probabilities
y_pred = np.round(y_pred_probs)  # Convert probabilities to binary predictions (0 or 1)

# Ensure y_test is a numpy array
y_test = np.array(y_test)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((y_pred - y_test.reshape(-1, 1)) ** 2))
print(f'RMSE: {rmse}') #0.6863381227311184
# Test Accuracy: 0.5426747798919678
# Test Loss (Binary Crossentropy): 0.65814608335495
# RMSE: 0.6762582523714945
# Save the model in native Keras format
model.save('models/sentiment_model.keras')

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open('models/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

    