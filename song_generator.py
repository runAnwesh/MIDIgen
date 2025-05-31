# at top of songgenerator.py
import pretty_midi
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae.configs import CONFIG_MAP


# choose a pretrained MusicVAE checkpoint:
config = CONFIG_MAP['cat-mel_2bar_big']    # 2-bar melodies, “cat-mel” interpolation
music_vae = TrainedModel(
    config, batch_size=4,
    checkpoint_dir_or_path='path/to/cat-mel_2bar_big.tar')


def generate_ai_clip(length_bars=8, temperature=1.0):
    """Returns a PrettyMIDI object of AI-generated melody."""
    # z & decoder_samples → a NoteSequence
    ns = music_vae.sample(n=1, length=length_bars * config.hparams.max_seq_len // 2,  
                          temperature=temperature)[0]
    # Convert NoteSequence → PrettyMIDI
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.instruments[0].notes = pretty_midi.NoteSequenceToMidi(ns).instruments[0].notes
    return pm

def generate_songs(self):
        key = self.key_var.get()
        scale = scales[key]

        for i, (_, song_name_entry) in enumerate(self.lines):
            song_name = song_name_entry.get().strip()
            if not song_name:
                continue

            midi = MIDIFile(1)
            song_structure = song_structures[i % len(song_structures)]  # Cycle through song structures

            midi.addTrackName(track=0, time=0, trackName=song_name)
            midi.addTempo(track=0, time=0, tempo=120)

            time = 0

            for section_name in song_structure:
                section_chords = random.choice(chord_progressions[section_name])
                for chord_name in section_chords:
                    chord_notes = get_chord_notes(chord_name[0], key, scales[chord_name[1]])
                    for note in chord_notes:
                        midi.addNote(
                            track=0,
                            channel=0,
                            pitch=note,
                            time=time,
                            duration=1,
                            volume=100
                        )
                    time += 1

            filename = f"{song_name.replace(' ', '_').lower()}.mid"
            with open(filename, "wb") as output_file:
                midi.writeFile(output_file)