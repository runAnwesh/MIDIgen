import os
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq

# --- Model Configuration ---
# This dictionary holds various pretrained models you can use.
# 'cat-trio_16bar' is a good general-purpose model for generating 16-bar, 3-part arrangements (melody, bass, drums).
# You can experiment with other models from the Magenta documentation.
PRETRAINED_MODELS = {
    "trio_16_bar": "cat-trio_16bar"
}

# --- Model Loading ---
# To save resources, we'll load the model once when the server starts.
# This avoids reloading the model for every request.
MODEL_NAME = "trio_16_bar"
config = configs.CONFIG_MAP[PRETRAINED_MODELS[MODEL_NAME]]
# The model will be downloaded from Google's servers the first time this runs.
# It will be cached locally for subsequent runs.
trio_model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=f"gs://magentadata/models/music_vae/colab2/{PRETRAINED_MODELS[MODEL_NAME]}.ckpt")


def generate_midi_sequence(output_dir: str = "generated_midi") -> str:
    """
    Generates a new multi-instrument MIDI sequence using the pre-loaded MusicVAE model.

    This function generates a number of sequences and selects the first one.
    The generated MIDI data is saved to a file.

    Args:
        output_dir: The directory where the generated MIDI file will be saved.

    Returns:
        The file path of the generated MIDI file.
    """
    try:
        # Generate 4 sequences. You can adjust this number.
        # The 'temperature' parameter controls the randomness of the output.
        # Higher values (e.g., 1.0) are more random, while lower values (e.g., 0.5) are more conservative.
        generated_sequences = trio_model.sample(n=4, length=256, temperature=0.9)

        if not generated_sequences:
            raise Exception("Model did not generate any sequences.")

        # Select the first generated sequence
        sequence = generated_sequences[0]

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a unique filename
        output_path = os.path.join(output_dir, "generated_pattern.mid")

        # Save the sequence as a MIDI file
        note_seq.sequence_proto_to_midi_file(sequence, output_path)

        return output_path

    except Exception as e:
        print(f"An error occurred during MIDI generation: {e}")
        return ""

if __name__ == '__main__':
    # This is an example of how to use the function.
    # When you integrate this with a web framework, you will call
    # generate_midi_sequence() from within your API endpoint.
    midi_file_path = generate_midi_sequence()

    if midi_file_path:
        print(f"Successfully generated MIDI file at: {midi_file_path}")
    else:
        print("Failed to generate MIDI file.")