#!/bin/bash
# Environment setup script for MIDIgen based on test.ipynb

# Install system dependencies
sudo apt-get update -qq
sudo apt-get install -qq libfluidsynth2 fluid-soundfont-gm build-essential libasound2-dev libjack-dev

# Install Python dependencies
pip install --upgrade pip
pip install pyfluidsynth magenta tensorflow numpy

# Note: 'google.colab' and 'files' are only available in Colab, not needed locally.
# 'magenta.music', 'magenta.models.music_vae', etc. are part of the 'magenta' pip package. 