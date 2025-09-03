# Real-Time Audio Spectrogram
#
# This script captures audio from the default microphone, computes a spectrogram
# in real-time, and displays it using matplotlib.
#
# Dependencies:
# - PyAudio: For audio input. Install with `pip install pyaudio`.
#   - On macOS, you might need to install portaudio first: `brew install portaudio`
#   - On Debian/Ubuntu: `sudo apt-get install python3-pyaudio portaudio19-dev`
# - numpy: For numerical operations. Install with `pip install numpy`.
# - matplotlib: For plotting. Install with `pip install matplotlib`.

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
# Audio settings
FORMAT = pyaudio.paInt16  # Audio format (16-bit integer)
CHANNELS = 1             # Mono audio
RATE = 44100             # Sample rate (Hz)
CHUNK = 1024 * 2         # Number of audio frames per buffer

# Spectrogram settings
N_FFT = 1024             # Number of FFT points
WINDOW = np.hanning(CHUNK) # Hanning window to reduce spectral leakage
HISTORY_LEN = 100        # Number of time frames to display in the spectrogram

# --- Initialization ---
# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream from default input device
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 5))

# Create the initial spectrogram data array
# The frequency axis will have N_FFT/2 + 1 bins
spectrogram_data = np.zeros((int(N_FFT / 2) + 1, HISTORY_LEN))

# Create the image plot. 'viridis' is a good colormap for spectrograms.
# The y-axis represents frequency, and the x-axis represents time.
im = ax.imshow(
    spectrogram_data,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='none',
    extent=[0, HISTORY_LEN, 0, RATE / 2 / 1000] # Display frequency in kHz
)

# Add labels and a color bar
ax.set_xlabel('Time Frames')
ax.set_ylabel('Frequency (kHz)')
ax.set_title('Real-Time Audio Spectrogram')
fig.colorbar(im, ax=ax, format='%+2.0f dB', label='Magnitude')

print("Starting audio stream...")

def update_plot(frame):
    """
    This function is called by FuncAnimation for each new frame.
    It reads audio data, computes the FFT, and updates the spectrogram plot.
    """
    global spectrogram_data
    
    try:
        # Read a chunk of audio data from the stream
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        # Convert the raw byte data to a numpy array of integers
        data = np.frombuffer(raw_data, dtype=np.int16)

        # Apply the window function to the data
        windowed_data = data * WINDOW

        # Compute the Fast Fourier Transform (FFT)
        fft_data = np.fft.rfft(windowed_data, n=N_FFT)
        
        # Calculate the magnitude spectrum in decibels (dB)
        # We add a small epsilon to avoid log(0) errors
        magnitude = 20 * np.log10(np.abs(fft_data) + 1e-10)
        
        # --- Update Spectrogram Data ---
        # Shift the existing spectrogram data one time frame to the left
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        
        # Add the new magnitude spectrum to the rightmost column
        spectrogram_data[:, -1] = magnitude

        # Update the image data
        im.set_array(spectrogram_data)
        
        # Update the color scale (clim) to fit the new data
        im.set_clim(vmin=np.min(spectrogram_data), vmax=np.max(spectrogram_data))

    except Exception as e:
        print(f"An error occurred: {e}")
        
    return [im]

# Create the animation
# The interval controls how often the plot updates in milliseconds.
ani = animation.FuncAnimation(
    fig,
    update_plot,
    blit=True,
    interval=10 # A smaller interval leads to a smoother, more real-time feel
)

# Show the plot
plt.show()

# --- Cleanup ---
print("Stopping audio stream...")
stream.stop_stream()
stream.close()
p.terminate()
