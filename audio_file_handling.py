from pydub import AudioSegment    
from pydub.playback import play
import os
import random

# Set the audio player to afplay (macOS)
AudioSegment.converter = "/usr/bin/afplay"

def listen_to_audio_file_with_chuncking (audio_file, number_of_chunks=9):
    # Play the audio
    print("Playing the whole audio...")
    play(audio_file)
    # Split the audio into 9 (number_of_chunks) equal parts
    chunks = [audio_file[i * len(audio_file) // number_of_chunks: (i + 1) * len(audio_file) // number_of_chunks] for i in range(number_of_chunks)]
    print("Chunking file into ", number_of_chunks, "chunks of equal size...")
    print("Now playing chucks of the same audio one at a time...")
    # Iterate through and play each chunk when you press Enter
    for i, chunk in enumerate(chunks, start=1):
        input("Press Enter to play chunk {}...".format(i))
        play(chunk)

# Specify the file path
file_path = ''

# Load the audio file
audio_file = AudioSegment.from_file(file_path, format="wav")
listen_to_audio_file_with_chuncking(audio_file, 9)

# Specify the directory path
directory_path = 'assets/speech_commands_v0.01/eight/'

# Get a list of audio files in the directory
audio_files = [file for file in os.listdir(directory_path) if file.endswith('.wav')]

print("\n" + "$" * 80 + "\n")
print("\n", "Now calculating average values for directory eight...", "\n")

# Initialize lists to store audio durations
durations = []

# Iterate through the audio files and calculate durations
for file in audio_files:
    audio_file_path = os.path.join(directory_path, file)
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    duration_in_seconds = len(audio) / 1000  # Convert to seconds
    durations.append(duration_in_seconds)

# Calculate minimum, maximum, and average duration
if durations:
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = sum(durations) / len(durations)
    print("Size of durations:", len(durations))
    print(f"Minimum Duration: {min_duration:.2f} seconds")
    print(f"Maximum Duration: {max_duration:.2f} seconds")
    print(f"Average Duration: {avg_duration:.2f} seconds")
else:
    print("No audio files found.")

print("\n" + "%" * 80 + "\n")

print("\n", "Now picking random files until you end the sequence...", "\n")

while True:
# Select a random audio file from the list
    random_audio_file_name = random.choice(audio_files)
    print("\n" + "#" * 80 + "\n")
    print(random_audio_file_name, "\n")
    file_path = directory_path + random_audio_file_name
    random_audio_file = AudioSegment.from_file(file_path, format="wav")
    listen_to_audio_file_with_chuncking(random_audio_file)
    print("\n", "\n")
    user_input = input("Press Enter or 'y' to hear another random audio file, or 'n' to quit: ")
    if user_input.lower() in ('', 'y'):
        # Continue the loop
        pass
    elif user_input.lower() == 'n' or user_input.lower() == 'q':
        # Exit the loop
        break
    else:
        # Handle invalid input
        print("Invalid input. Please press Enter or 'y' to continue. Otherwise 'n' or 'q' to quit.")




