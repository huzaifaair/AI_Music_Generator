import os
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding  
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import ModelCheckpoint 

# Function to build the LSTM model
def build_model(vocab_size, sequence_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=sequence_length),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main script execution
if __name__ == "__main__":
    # Paths and parameters
    data_path = "./data/processed_data/sequences.npy"  # Path to preprocessed sequences
    model_save_path = "./models/music_generation_model.h5"  # Path to save the trained model

    # Load the preprocessed data
    print("Loading preprocessed data...")
    sequences = np.load(data_path)

    # Prepare data for training
    print("Preparing data for training...")
    vocab_size = 128  # Number of unique pitches (MIDI note range)
    X = sequences[:, :-1]  # Input sequences
    y = sequences[:, -1]  # Target notes

    y = to_categorical(y, num_classes=vocab_size)  # One-hot encode the target notes

    sequence_length = X.shape[1]

    # Build the model
    print("Building the model...")
    model = build_model(vocab_size, sequence_length)

    # Setup checkpoints to save the best model
    checkpoint = ModelCheckpoint(
    filepath="./models/music_generation_model.keras",  # Change the file extension to .keras
    monitor="loss",
    save_best_only=True,
    mode="min"
)

    # Train the model
    print("Training the model...")
    model.fit(
        X, y, epochs=50, batch_size=64, callbacks=[checkpoint], verbose=1
    )

    print(f"Model training complete. Best model saved to {model_save_path}.")
