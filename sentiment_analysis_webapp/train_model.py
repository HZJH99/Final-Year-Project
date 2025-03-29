# Import necessary libraries
import pandas as pd
import pickle
import numpy as np
import nltk
import tensorflow as tf
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset and drop rows with missing reviews
data = pd.read_csv('amazon_fine_food_reviews.csv')
data = data[['Text', 'Score']]
data.dropna(subset=['Text'], inplace=True)

# Assign soft sentiment labels based on original score
def assign_soft_label(score):
    if score <= 2:
        return np.random.uniform(0.05, 0.3)  # Negative
    elif score == 3:
        return np.random.uniform(0.45, 0.55)  # Neutral
    else:
        return np.random.uniform(0.7, 0.95) # Positive

# Apply soft labels
data['Sentiment'] = data['Score'].apply(assign_soft_label)

# Initialize lemmatizer and stop words (excluding negations)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'} 

# Clean and normalize the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\b(can't|won't|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|n't)\b", "not", text)
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply the preprocessing function
data['Cleaned_Text'] = data['Text'].apply(preprocess_text)

# Convert text to TF-IDF features (bigrams, max 3000 features)
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
x = vectorizer.fit_transform(data['Cleaned_Text']).toarray().astype('float32')
y = data['Sentiment']

# Save the TF-IDF vectorizer for future use
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


# Convert to tf.data.Dataset for efficient data loading
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(256).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256).prefetch(tf.data.AUTOTUNE)

# Build the model using Dense layers, dropout, and LeakyReLU activations
final_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, kernel_regularizer=l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(64, kernel_regularizer=l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(1, activation='linear') # Linear output for regression-style soft sentiment score
])

# Compile model with Huber loss (robust for regression) and MSE as metric
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(), 
    metrics=['mse']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)

# Train the model 
final_model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the trained model
final_model.save('sentiment_model.h5')
print("Final Model saved successfully!")