import os
import copy
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Magenta and note_seq imports ---
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq

# ==============================================================================
# ALL THE MIDI GENERATION LOGIC FROM THE PREVIOUS STEP GOES HERE.
# This code is unchanged.
# ==============================================================================

# --- Model Configuration ---
PRETRAINED_MODELS = {
    "melody": "mel_2bar_big",
    "drums": "cat-drums_2bar_small"
}

# --- Model Loading Cache ---
LOADED_MODELS = {}

def get_model(model_name: str) -> TrainedModel:
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]
    config_name = PRETRAINED_MODELS[model_name]
    config = configs.CONFIG_MAP[config_name]
    checkpoint_path = f"gs://magentadata/models/music_vae/colab2/{config_name}.ckpt"
    model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=checkpoint_path)
    LOADED_MODELS[model_name] = model
    print(f"Loaded model: {model_name}")
    return model

# --- MIDI Post-Processing ---
def create_pad_from_sequence(sequence: note_seq.NoteSequence, note_length: float = 1.5) -> note_seq.NoteSequence:
    pad_sequence = copy.deepcopy(sequence)
    for note in pad_sequence.notes:
        note.end_time = note.start_time + note_length
    return pad_sequence

def filter_drum_pattern(sequence: note_seq.NoteSequence, allowed_pitches: set) -> note_seq.NoteSequence:
    filtered_sequence = note_seq.NoteSequence()
    filtered_sequence.tempos.add().qpm = sequence.tempos[0].qpm
    for note in sequence.notes:
        if note.pitch in allowed_pitches:
            filtered_sequence.notes.add().CopyFrom(note)
    filtered_sequence.total_time = sequence.total_time
    return filtered_sequence

# --- Main Generation Logic ---
def generate_instrument_midi(instrument_type: str, output_dir: str = "generated_midi") -> str:
    # This function is the same as before...
    try:
        sequence = None
        DRUM_PITCHES = {"kick": {35, 36}, "snare": {38, 40}, "closed_hat": {42, 44}, "open_hat": {46}, "clap": {39}}

        if instrument_type in ["lead", "pluck", "keys", "pad"]:
            model = get_model("melody")
            sequence = model.sample(n=1, temperature=0.8)[0]
        elif instrument_type in ["drums"] or instrument_type in DRUM_PITCHES:
            model = get_model("drums")
            sequence = model.sample(n=1, temperature=1.0)[0]
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}")

        final_sequence = sequence
        if instrument_type == "pad":
            final_sequence = create_pad_from_sequence(sequence)
        elif instrument_type in DRUM_PITCHES:
            final_sequence = filter_drum_pattern(sequence, DRUM_PITCHES[instrument_type])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{instrument_type}_pattern.mid")
        note_seq.sequence_proto_to_midi_file(final_sequence, output_path)
        return output_path
    except Exception as e:
        # Raise an exception that the API can catch
        raise RuntimeError(f"Error generating MIDI for {instrument_type}: {e}") from e

# ==============================================================================
# NEW FastAPI Application
# ==============================================================================
app = FastAPI()

# --- CORS Middleware ---
# This is crucial for allowing your Next.js app (running on localhost:3000)
# to communicate with your backend (running on localhost:8000).
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint Definition ---
VALID_INSTRUMENTS = [
    "lead", "pluck", "keys", "pad", "drums", 
    "kick", "snare", "closed_hat", "open_hat", "clap"
]

@app.get("/generate-midi")
def get_midi_endpoint(
    instrument: str = Query("lead", enum=VALID_INSTRUMENTS)
):
    """
    This endpoint generates and returns a MIDI file for the specified instrument.
    """
    try:
        print(f"Received request to generate MIDI for: {instrument}")
        midi_path = generate_instrument_midi(instrument)
        
        if not midi_path or not os.path.exists(midi_path):
            raise HTTPException(status_code=500, detail="Failed to generate MIDI file on the server.")
        
        # Return the file as a response
        return FileResponse(
            path=midi_path,
            media_type='audio/midi',
            filename=f"{instrument}_pattern.mid"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AI MIDI Generator Backend is running."}