import pyaudio
import wave
import threading

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 2

class AudioRecorder:
    """Handles audio stream + recording to WAV file."""

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False

    def start_stream(self):
        if self.stream is None:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
        return self.stream

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def terminate(self):
        self.p.terminate()

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        self.output_filename = None

    def start_recording(self, filename="output.wav"):
        """Start recording in a background thread."""
        self.frames = []
        self.recording = True
        self.output_filename = filename
        def record():
            while self.recording:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        threading.Thread(target=record, daemon=True).start()

    def stop_recording(self, filename=None):
        """Stop recording and save to file."""
        self.recording = False
        output_file = filename or self.output_filename or "output.wav"
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()
