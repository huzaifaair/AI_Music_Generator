import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from pretty_midi import PrettyMIDI, Instrument, Note 

# Function to generate a sequence of notes using the trained model
def generate_sequence(model, seed_sequence, num_notes_to_generate, sequence_length):
    generated_sequence = list(seed_sequence)

    for _ in range(num_notes_to_generate):
        input_sequence = np.array(generated_sequence[-sequence_length:]).reshape(1, sequence_length, 1)
        next_note = np.argmax(model.predict(input_sequence, verbose=0))
        generated_sequence.append(next_note)

    return generated_sequence

# Function to convert the generated sequence into a MIDI file
def sequence_to_midi(sequence, output_path):
    midi = PrettyMIDI()
    instrument = Instrument(program=0)  # Acoustic Grand Piano

    start_time = 0.0
    duration = 0.5  # Fixed duration for each note

    for pitch in sequence:
        note = Note(velocity=100, pitch=pitch, start=start_time, end=start_time + duration)
        instrument.notes.append(note)
        start_time += duration

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"MIDI file saved to {output_path}")

# Main script execution
if __name__ == "__main__":
    # Paths and parameters
    model_path = "./models/music_generation_model.keras"  # Corrected file extension for model
    seed_sequence_path = "./data/processed_data/sequences.npy"  # Path to the preprocessed sequences
    output_midi_path = "./generated_music/output.mid"  # Path to save the generated MIDI file

    # Load the trained model
    print("Loading trained model...")
    try:
        model = load_model(model_path)
    except OSError:
        print(f"Error: Model file not found at {model_path}. Make sure the file exists and is correctly named.")
        exit(1)

    # Load the seed sequence
    print("Loading seed sequence...")
    try:
        sequences = np.load(seed_sequence_path)
    except FileNotFoundError:
        print(f"Error: Seed sequence file not found at {seed_sequence_path}. Ensure the file exists.")
        exit(1)

    # Select a random seed sequence
    random_index = np.random.randint(0, len(sequences))
    seed_sequence = sequences[random_index, :-1]  # Use only input part of the sequence

    # Generate music
    print("Generating music...")
    num_notes_to_generate = 100  # Number of notes to generate
    sequence_length = seed_sequence.shape[0]
    generated_sequence = generate_sequence(model, seed_sequence, num_notes_to_generate, sequence_length)

    # Convert sequence to MIDI and save
    print("Converting sequence to MIDI...")
    sequence_to_midi(generated_sequence, output_midi_path)