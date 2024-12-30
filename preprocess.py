import os
import numpy as np
import pretty_midi 

# Function to extract notes from MIDI files
def extract_notes(midi_files_path):
    notes = []
    for file in os.listdir(midi_files_path):
        if file.endswith('.mid'):
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(midi_files_path, file))
                for instrument in midi.instruments:
                    if not instrument.is_drum:  # Ignore drum tracks
                        notes.extend([note.pitch for note in instrument.notes])
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return notes

# Function to create sequences of notes
def create_sequences(notes, sequence_length=50):
    sequences = []
    for i in range(len(notes) - sequence_length):
        seq = notes[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Main script execution
if __name__ == "__main__":
    # Define paths
    midi_files_path = "./data/midi_files"  # Path to MIDI files
    output_path = "./data/processed_data/sequences.npy"  # Path to save sequences

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Extract notes from MIDI files
    print("Extracting notes from MIDI files...")
    notes = extract_notes(midi_files_path)

    if len(notes) == 0:
        print("No notes were extracted. Please check your MIDI files.")
        exit()

    # Step 2: Create sequences
    print("Creating sequences...")
    sequence_length = 50  # length of the sequences
    sequences = create_sequences(notes, sequence_length)

    # Step 3: Save sequences to a .npy file
    print(f"Saving sequences to {output_path}...")
    np.save(output_path, sequences)

    print(f"Successfully saved {len(sequences)} sequences.")
