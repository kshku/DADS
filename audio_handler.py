"""
DADS - Stutter Detection System
File: audio_handler.py
Role: Manages all audio *input* (recording) logic.
"""

from PyQt5.QtCore import QObject, pyqtSignal, QBuffer, QIODevice
from PyQt5.QtMultimedia import QAudioInput, QAudioFormat, QAudioDeviceInfo

class AudioHandler(QObject):
    """
    Handles audio recording using QAudioInput.
    Emits the raw audio data when recording is finished.
    """
    
    # Signal: (raw_audio_bytes)
    recording_finished = pyqtSignal(bytes)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.audio_input = None
        self.audio_buffer = None
        
        # --- Recording Format ---
        self.format = QAudioFormat()
        self.format.setSampleRate(44100)
        self.format.setChannelCount(1)
        self.format.setSampleSize(16)
        self.format.setCodec("audio/pcm")
        self.format.setByteOrder(QAudioFormat.LittleEndian)
        self.format.setSampleType(QAudioFormat.SignedInt)
        
        # Check if format is supported
        try:
            info = QAudioDeviceInfo.defaultInputDevice()
            if not info.isFormatSupported(self.format):
                print("Default format not supported, trying nearest.")
                self.format = info.nearestFormat(self.format)
        except Exception as e:
            print(f"Error initializing audio device: {e}. Using default format.")
            # Continue with default 44.1kHz, it may work.

    def get_sample_rate(self):
        """Returns the sample rate of the recording."""
        return self.format.sampleRate()

    def start_recording(self):
        """
        Initializes and starts the audio input.
        """
        if self.audio_input:
            self.stop_recording() # Stop any previous
            
        self.audio_input = QAudioInput(self.format, self)
        self.audio_buffer = QBuffer()
        self.audio_buffer.open(QIODevice.ReadWrite)
        
        # Start piping audio from mic to buffer
        self.audio_input.start(self.audio_buffer)

    def stop_recording(self):
        """
        Stops the audio input and emits the data.
        """
        if self.audio_input:
            self.audio_input.stop()
            
            # Get data from the buffer
            data = self.audio_buffer.data()
            
            # Clean up
            self.audio_buffer.close()
            self.audio_buffer = None
            self.audio_input = None
            
            # Emit the signal with the raw data
            self.recording_finished.emit(data)
