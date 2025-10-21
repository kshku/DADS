import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from audio_utils import AudioRecorder, CHUNK, RATE

# Mel filterbank helper
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)

def mel_filterbank(n_filters=40, n_fft=1024, sr=44100):
    """Generate mel filterbank matrix."""
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, center):
            fbanks[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbanks[i - 1, j] = (right - j) / (right - center)
    return fbanks

# --- Spectrogram Runner ---
def run_mel_spectrogram(stop_event):
    recorder = AudioRecorder()
    stream = recorder.start_stream()

    N_FFT = 1024
    HISTORY_LEN = 100
    DPI = 100  # Adjust DPI for better resolution
    mel_fb = mel_filterbank(n_filters=40, n_fft=N_FFT, sr=RATE)

    # Data storage
    mel_spec = np.zeros((mel_fb.shape[0], HISTORY_LEN))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)  # Increased figure size for complete zoom-out
    im = ax.imshow(
        mel_spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="none",
        extent=[0, HISTORY_LEN, 0, mel_fb.shape[0]]
    )
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Filter Bank")
    ax.set_title("Real-Time Mel Spectrogram")
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")

    def update(frame):
        nonlocal mel_spec
        if stop_event.is_set():
            plt.close(fig)
            return [im]
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(raw_data, dtype=np.int16)

            # FFT
            fft_data = np.fft.rfft(data, n=N_FFT)
            power_spectrum = np.abs(fft_data) ** 2

            # Apply mel filterbank
            mel_energy = np.dot(mel_fb, power_spectrum)
            mel_db = 20 * np.log10(mel_energy + 1e-10)

            # Update spectrogram
            mel_spec = np.roll(mel_spec, -1, axis=1)
            mel_spec[:, -1] = mel_db
            im.set_array(mel_spec)
            im.set_clim(vmin=np.min(mel_spec), vmax=np.max(mel_spec))
        except Exception as e:
            print(f"Error: {e}")
        return [im]

    ani = animation.FuncAnimation(fig, update, blit=True, interval=30)
    plt.show()

    recorder.stop_stream()
    recorder.terminate()
