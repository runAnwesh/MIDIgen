import os
import copy
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq

# --- Model Configuration ---
# We now define multiple models for different tasks.
# 'mel_2bar_big' is great for short, catchy monophonic melodies (leads, plucks).
# 'cat-drums_2bar_small' is a model trained specifically on drum patterns.
PRETRAINED_MODELS = {
    "melody": "mel_2bar_big",
    "drums": "cat-drums_2bar_small"
}

# --- Model Loading ---
# To save resources, we'll use a dictionary to cache the loaded models.
# This prevents reloading from disk or re-downloading for every request.
LOADED_MODELS = {}

def get_model(model_name: str) -> TrainedModel:
    """Loads a model from the cache or initializes it if not present."""
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    config_name = PRETRAINED_MODELS[model_name]
    config = configs.CONFIG_MAP[config_name]
    
    # The model will be downloaded from Google's servers the first time.
    checkpoint_path = f"gs://magentadata/models/music_vae/colab2/{config_name}.ckpt"
    model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=checkpoint_path)
    
    LOADED_MODELS[model_name] = model
    print(f"Loaded model: {model_name}")
    return model

# --- MIDI Post-Processing ---

def create_pad_from_sequence(sequence: note_seq.NoteSequence, note_length: float = 1.5) -> note_seq.NoteSequence:
    """Converts a sequence to a pad-like style by extending note durations."""
    pad_sequence = copy.deepcopy(sequence)
    for note in pad_sequence.notes:
        note.end_time = note.start_time + note_length
    return pad_sequence

def filter_drum_pattern(sequence: note_seq.NoteSequence, allowed_pitches: set) -> note_seq.NoteSequence:
    """Filters a drum sequence to only include specified drum parts (pitches)."""
    filtered_sequence = note_seq.NoteSequence()
    filtered_sequence.tempos.add().qpm = sequence.tempos[0].qpm
    
    for note in sequence.notes:
        if note.pitch in allowed_pitches:
            filtered_sequence.notes.add().CopyFrom(note)
            
    filtered_sequence.total_time = sequence.total_time
    return filtered_sequence

# --- Main Generation Logic ---

def generate_instrument_midi(instrument_type: str, output_dir: str = "generated_midi") -> str:
    """
    Generates a MIDI file tailored to a specific instrument type.

    Args:
        instrument_type: One of ['lead', 'pluck', 'keys', 'pad', 'drums', 'kick', 'snare', 'closed_hat'].
        output_dir: The directory to save the MIDI file.

    Returns:
        The file path of the generated MIDI file.
    """
    try:
        sequence = None
        
        # General MIDI Drum Map Pitches
        DRUM_PITCHES = {
            "kick": {35, 36},
            "snare": {38, 40},
            "closed_hat": {42, 44},
            "open_hat": {46},
            "clap": {39}
        }

        # 1. Select model and generate base sequence
        if instrument_type in ["lead", "pluck", "keys", "pad"]:
            model = get_model("melody")
            # Generate a single sequence with medium randomness
            sequence = model.sample(n=1, temperature=0.8)[0]

        elif instrument_type in ["drums"] or instrument_type in DRUM_PITCHES:
            model = get_model("drums")
            sequence = model.sample(n=1, temperature=1.0)[0]
        
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}")

        # 2. Post-process the sequence based on instrument type
        final_sequence = sequence
        if instrument_type == "pad":
            final_sequence = create_pad_from_sequence(sequence)
        
        elif instrument_type in DRUM_PITCHES:
            final_sequence = filter_drum_pattern(sequence, DRUM_PITCHES[instrument_type])


        # 3. Save the final sequence to a MIDI file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{instrument_type}_pattern.mid")
        note_seq.sequence_proto_to_midi_file(final_sequence, output_path)
        
        return output_path

    except Exception as e:
        print(f"An error occurred during MIDI generation for {instrument_type}: {e}")
        return ""


if __name__ == '__main__':
    print("Generating MIDI for various instruments...")
    
    # Example usage:
    instruments_to_generate = [
        "lead", 
        "pad", 
        "drums", 
        "kick", 
        "snare",
        "closed_hat"
    ]
    
    for instrument in instruments_to_generate:
        midi_file = generate_instrument_midi(instrument)
        if midi_file:
            print(f"✓ Successfully generated: {midi_file}")
        else:
            print(f"✗ Failed to generate for: {instrument}")