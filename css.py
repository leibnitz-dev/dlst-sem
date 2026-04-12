import os
from gtts import gTTS
from pydub import AudioSegment

# Create folder for audio files
if not os.path.exists("audio"):
    os.makedirs("audio")

# Common kiosk words
words = [
    "welcome", "please", "select", "enter", "ticket",
    "payment", "successful", "cancel", "exit",
    "thank you", "number", "confirm", "back",
    "next", "invalid", "try again"
]

for word in words:

    print("Generating:", word)

    # Generate speech
    tts = gTTS(text=word, lang="en")

    # Temporary mp3 file
    tts.save(f"audio/{word}.mp3")




base_dir = os.path.dirname(__file__)
audio_folder = os.path.join(base_dir, "audio")

files = ["welcome.mp3", "please.mp3", "select.mp3", "confirm.mp3", "ticket.mp3"]
files = [os.path.join(audio_folder, f) for f in files]

with open("output.mp3", "wb") as outfile:
    for fname in files:
        with open(fname, "rb") as infile:
            outfile.write(infile.read())
