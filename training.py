import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, TextVectorization, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import classifier

PATH = './dataset/IMDB Dataset.csv'
MAX_SEQUENCE_LENGTH = 500
MAX_TOKENS = 20000
EMBEDDING_DIM = 64
MODEL_PATH = './model/classifier.keras'
TOKENIZER_PATH = './model/tokenizer.pkl'


def save_tokenizer(tokenizer):
    with open(TOKENIZER_PATH, 'wb') as f:
        config = tokenizer.get_config()
        config['vocabulary'] = tokenizer.get_vocabulary()
        pickle.dump(config, f)


def main():
    print("Loading and Preprocessing Data...")
    df = classifier.loading_and_preprocessing()
    
    texts = df['review'].values
    labels = df['feedback'].values

    print("Vectorizing...")

    vectorizer = TextVectorization(max_tokens=MAX_TOKENS,
                                   output_sequence_length=MAX_SEQUENCE_LENGTH)
                                   
    vectorizer.adapt(texts)

    save_tokenizer(vectorizer)

    vocab_size = len(vectorizer.get_vocabulary())
    print(f"Vocabulary Size: {vocab_size}")
    
    X = classifier.data_preparation(texts, vectorizer)
    X = np.array(X)
    y = np.array(labels)
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    print("Building Model (Bidirectional)...")
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Dropout(0.3)) 
    
    model.add(Bidirectional(LSTM(64))) 
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    

    print("Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )
    model.save(MODEL_PATH)
    print("Model Saved !")
    print("Training Complete.")
    
if __name__ == "__main__":
    main()
