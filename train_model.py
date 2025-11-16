import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
RANDOM_SEED = 42
VOCAB_SIZE = 10000  # Max number of unique words to use
MAX_SEQUENCE_LENGTH = 250 # Max length of a review sequence (padding/truncating)
EMBEDDING_DIM = 100 # Dimension of the word vectors
BATCH_SIZE = 32
EPOCHS = 10

# --- 1. Load Data and Preprocessing ---
print("1. Loading and preparing data...")
try:
    # Use the larger IMDB_Dataset.csv if available
    df = pd.read_csv('IMDB_Dataset.csv').sample(n=10000, random_state=RANDOM_SEED) 
except FileNotFoundError:
    print("IMDB_Dataset.csv not found. Using a small sample for demonstration.")
    data = {
        'review': [
            "This movie was absolutely fantastic and a masterpiece.",
            "The acting was horrible and the plot was confusing.",
            "A decent effort, but nothing special.",
            "I loved every minute of this film. Highly recommended!",
            "Avoid this at all costs, a total waste of time."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative']
    }
    df = pd.DataFrame(data)

# Convert sentiment to numerical labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Simple Preprocessing (Keep stopwords for CNN, but clean HTML/chars)
lemmatizer = WordNetLemmatizer()
english_stop_words = set(stopwords.words('english')) # Note: Stopword removal often hurts CNN performance, but we keep the func clean.

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # For CNN, we only tokenize after this stage, we don't remove stopwords or lemmatize here.
    return text

df['clean_review'] = df['review'].apply(preprocess_text)

X = df['clean_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# --- 2. Feature Extraction: Tokenization and Embedding ---
print("2. Tokenizing and Padding Sequences...")

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# --- 3. Build the CNN Model ---
print("3. Building CNN Model...")

model = Sequential([
    # Layer 1: Embedding Layer - Converts integer indices to dense vectors
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    
    # Layer 2: Convolutional Layer - Extracts local features (n-grams)
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    
    # Layer 3: Global Max Pooling - Downsamples and finds the most important feature per filter
    GlobalMaxPooling1D(),
    
    # Layer 4: Dense Layer - Standard fully connected layer for high-level features
    Dense(64, activation='relu'),
    
    # Layer 5: Output Layer - Sigmoid activation for binary classification (Positive/Negative)
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- 4. Model Training and Saving ---
print("4. Training Model...")

# EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_padded, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_padded, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# --- 5. Save the trained model and tokenizer ---
print("\n5. Evaluating and Saving Files...")

# Save the trained Keras model
model.save('cnn_sentiment_model.h5')

# Save the tokenizer (crucial for preprocessing new, live input)
with open('cnn_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ CNN Model and Tokenizer saved successfully.")
