import os
import copy
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq
from note_seq.sequences_lib import adjust_notes_to_new_tempo

# ==============================================================================
# 1. NEW CONFIGURATION: GENRE-TO-MODEL MAPPING
# This is the core of the new system. We map a genre to the specific
# Magenta model checkpoints that best suit its style.
# ==============================================================================
GENRE_MODEL_MAP = {
    "pop": {
        "melody_model": "mel_4bar_med_q2",
        "drum_model": "cat-drums_2bar_small"
    },
    "hiphop": {
        "melody_model": "mel_2bar_big",
        "drum_model": "groovae_4bar" # GrooVAE is excellent for hip-hop drums
    },
    "dance": {
        "melody_model": "mel_2bar_big",
        "drum_model": "groovae_4bar"
    },
    "cinematic": {
        "melody_model": "mel_16bar_big_q2", # A model for longer, more complex melodies
        "drum_model": None # Cinematic music might not have drums
    },
    
}

VALID_GENRES = list(GENRE_MODEL_MAP.keys())
VALID_INSTRUMENTS = ["lead", "pluck", "keys", "pad", "drums", "kick", "snare", "closed_hat", "open_hat", "clap"]


# ==============================================================================
# 2. CORE LOGIC (RESTRUCTURED FOR FLEXIBILITY)
# ==============================================================================

# --- Model Loading Cache ---
LOADED_MODELS = {}

def get_model(model_checkpoint_name: str) -> TrainedModel:
    """Loads a model by its specific checkpoint name."""
    if not model_checkpoint_name:
        return None
    if model_checkpoint_name in LOADED_MODELS:
        return LOADED_MODELS[model_checkpoint_name]
    
    try:
        config = configs.CONFIG_MAP[model_checkpoint_name]
        model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=f"gs://magentadata/models/music_vae/colab2/{model_checkpoint_name}.ckpt")
        LOADED_MODELS[model_checkpoint_name] = model
        print(f"Loaded model: {model_checkpoint_name}")
        return model
    except (KeyError, IOError) as e:
        print(f"Could not load model {model_checkpoint_name}: {e}")
        # In a real app, you might fall back to a default model here
        raise HTTPException(status_code=500, detail=f"AI model '{model_checkpoint_name}' not found or could not be loaded.")


# --- Post-Processing Helpers (Unchanged) ---
def create_pad_from_sequence(sequence: note_seq.NoteSequence, note_length: float = 2.0) -> note_seq.NoteSequence:
    # (Code from previous step is unchanged)
    pad_sequence = copy.deepcopy(sequence)
    for note in pad_sequence.notes: note.end_time = note.start_time + note_length
    return pad_sequence

def filter_drum_pattern(sequence: note_seq.NoteSequence, allowed_pitches: set) -> note_seq.NoteSequence:
    # (Code from previous step is unchanged)
    filtered_sequence = note_seq.NoteSequence()
    if sequence.tempos: filtered_sequence.tempos.add().qpm = sequence.tempos[0].qpm
    for note in sequence.notes:
        if note.pitch in allowed_pitches: filtered_sequence.notes.add().CopyFrom(note)
    if sequence.notes: filtered_sequence.total_time = max(n.end_time for n in sequence.notes)
    return filtered_sequence


# --- Main Orchestration Function ---
def generate_midi_pattern(instrument: str, genre: str, bpm: int, output_dir: str = "generated_midi") -> str:
    """Generates a single instrument track influenced by genre and set to a specific BPM."""
    
    # 1. Select the right model based on genre and instrument type
    genre_map = GENRE_MODEL_MAP.get(genre)
    if not genre_map:
        raise HTTPException(status_code=400, detail=f"Invalid genre: {genre}")

    is_drum_instrument = instrument in ["drums"] or instrument in {"kick", "snare", "closed_hat", "open_hat", "clap"}
    
    model_checkpoint = genre_map["drum_model"] if is_drum_instrument else genre_map["melody_model"]
    if not model_checkpoint:
        raise HTTPException(status_code=400, detail=f"The genre '{genre}' does not support the instrument type '{instrument}'.")
    
    model = get_model(model_checkpoint)

    # 2. Generate the base sequence
    # Temperature can also be a parameter later! Lower = more predictable, Higher = more random.
    base_sequence = model.sample(n=1, temperature=0.9)[0]

    # 3. Apply instrument-specific post-processing
    final_sequence = base_sequence
    DRUM_PITCHES = {"kick": {35, 36}, "snare": {38, 40}, "closed_hat": {42, 44}, "open_hat": {46}, "clap": {39}}
    if instrument == "pad":
        final_sequence = create_pad_from_sequence(base_sequence)
    elif instrument in DRUM_PITCHES:
        final_sequence = filter_drum_pattern(base_sequence, DRUM_PITCHES[instrument])
    
    # 4. Adjust the tempo to the user's specified BPM
    final_sequence = adjust_notes_to_new_tempo(final_sequence, new_qpm=bpm)

    # 5. Save the final MIDI file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{genre}_{instrument}_{bpm}bpm.mid")
    note_seq.sequence_proto_to_midi_file(final_sequence, output_path)
    return output_path


# ==============================================================================
# 3. FastAPI Application Setup (Updated Endpoint)
# ==============================================================================
app = FastAPI()

origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/generate-midi")
def generate_midi_endpoint(
    instrument: str = Query("lead", enum=VALID_INSTRUMENTS),
    genre: str = Query("pop", enum=VALID_GENRES),
    bpm: int = Query(120, ge=40, le=240) # BPM with a reasonable range
):
    """
    Generates a MIDI pattern for a specific instrument,
    influenced by a genre and set to a custom BPM.
    """
    try:
        print(f"Request: Instrument={instrument}, Genre={genre}, BPM={bpm}")
        midi_path = generate_midi_pattern(instrument=instrument, genre=genre, bpm=bpm)
        return FileResponse(
            path=midi_path,
            media_type='audio/midi',
            filename=os.path.basename(midi_path)
        )
    except Exception as e:
        # Catch exceptions from model loading or generation
        if isinstance(e, HTTPException):
            raise e
        print(f"An internal error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AI MIDI Generator Backend v2 is running."}