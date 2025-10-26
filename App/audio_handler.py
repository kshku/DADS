"""
Audio Handler Module
Handles audio recording, playback, and file I/O operations
"""
import wave
import io
from PyQt5.QtCore import QObject, QIODevice, QBuffer, QByteArray, QFile, QDataStream, QTimer
from PyQt5.QtMultimedia import (
    QAudioFormat, QAudioInput, QAudioDeviceInfo,
    QAudioOutput, QAudio
)

class AudioHandler(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_input = None
        self.audio_buffer = None
        self.audio_data = QByteArray()
        self.current_sample_rate = 0
        self.audio_format_out = QAudioFormat()
        self.audio_output = None
        self.audio_play_buffer = QBuffer()
        self.bytes_per_second = 0
        self.playback_start_time = 0
        
        self.init_audio_format()
    
    def init_audio_format(self):
        """Initialize audio format for recording and playback"""
        format = QAudioFormat()
        format.setSampleRate(16000)  # Set to 16kHz
        format.setChannelCount(1)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)
        self.audio_format_out = format
        
        info = QAudioDeviceInfo.defaultInputDevice()
        if not info.isFormatSupported(format):
            print("Audio input format not supported by device.")
            return False
            
        self.audio_input = QAudioInput(info, format, self)
        self.audio_buffer = QBuffer()
        return True
    
    def init_audio_output(self):
        """Initialize audio output for playback"""
        info = QAudioDeviceInfo.defaultOutputDevice()
        
        # Try to use the exact format
        if not info.isFormatSupported(self.audio_format_out):
            print("Exact audio format not supported, trying to find nearest...")
            nearest = info.nearestFormat(self.audio_format_out)
            if nearest.sampleRate() != self.audio_format_out.sampleRate():
                print(f"Warning: Device using sample rate {nearest.sampleRate()} Hz instead of requested {self.audio_format_out.sampleRate()} Hz")
            self.audio_format_out = nearest
        
        self.audio_output = QAudioOutput(info, self.audio_format_out, self)
        self.audio_output.setBufferSize(32 * 1024)  # Smaller buffer for more responsive playback
        
        self.bytes_per_second = self.audio_format_out.sampleRate() * \
                                self.audio_format_out.channelCount() * \
                                (self.audio_format_out.sampleSize() // 8)
        return True
    
    def start_recording(self):
        """Start audio recording"""
        if not self.audio_buffer.open(QIODevice.WriteOnly):
            return False
        self.audio_data = QByteArray()
        self.current_sample_rate = 0
        self.audio_input.start(self.audio_buffer)
        return True
    
    def stop_recording(self):
        """Stop audio recording and store the data"""
        self.audio_input.stop()
        self.audio_buffer.close()
        self.audio_data = self.audio_buffer.data()
        self.current_sample_rate = self.audio_input.format().sampleRate()
        self.audio_buffer.setData(QByteArray())
        print(f"Recording stopped. {len(self.audio_data)} bytes captured.")
        return self.audio_data, self.current_sample_rate
    
    def load_wav_file(self, file_path):
        """Load audio from WAV file"""
        try:
            with io.BytesIO() as wav_bytes:
                # Read file
                with open(file_path, 'rb') as f:
                    wav_bytes.write(f.read())
                wav_bytes.seek(0)
                
                with wave.open(wav_bytes, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    samp_width = wav_file.getsampwidth()
                    self.current_sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    if samp_width != 2:
                        raise ValueError(f"Unsupported sample width: {samp_width}")
                    if n_channels != 1:
                        raise ValueError(f"Unsupported channel count: {n_channels}")
                    
                    pcm_data = wav_file.readframes(n_frames)
                    self.audio_data = QByteArray(pcm_data)
                    return True
        except Exception as e:
            print(f"Error loading WAV file: {e}")
            self.audio_data = QByteArray()
            self.current_sample_rate = 0
            return False
    
    def write_wav_file(self, file_device, pcm_data, audio_format):
        """Write WAV file with proper header"""
        data_len = len(pcm_data)
        stream = QDataStream(file_device)
        stream.setByteOrder(QDataStream.LittleEndian)
        
        # RIFF Header
        stream.writeRawData(b'RIFF')
        stream.writeInt32(36 + data_len)
        stream.writeRawData(b'WAVE')
        
        # Format Chunk
        stream.writeRawData(b'fmt ')
        stream.writeInt32(16)
        stream.writeInt16(1)
        
        num_channels = audio_format.channelCount()
        stream.writeInt16(num_channels)
        
        sample_rate = audio_format.sampleRate()
        stream.writeInt32(sample_rate)
        
        bits_per_sample = audio_format.sampleSize()
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        stream.writeInt32(byte_rate)
        
        block_align = num_channels * (bits_per_sample // 8)
        stream.writeInt16(block_align)
        stream.writeInt16(bits_per_sample)
        
        # Data Chunk
        stream.writeRawData(b'data')
        stream.writeInt32(data_len)
        file_device.write(pcm_data)
    
    def setup_playback(self):
        """Setup audio buffer for playback"""
        # Set the correct sample rate from the loaded audio
        self.audio_format_out.setSampleRate(self.current_sample_rate)
        
        if self.audio_output is not None:
            self.audio_output.stop()
            self.audio_output = None
        
        if not self.init_audio_output():
            return False
        
        # Ensure bytes_per_second is calculated
        self.bytes_per_second = self.audio_format_out.sampleRate() * \
                                self.audio_format_out.channelCount() * \
                                (self.audio_format_out.sampleSize() // 8)
                
        self.audio_play_buffer.close()
        self.audio_play_buffer.setData(self.audio_data)
        if not self.audio_play_buffer.open(QIODevice.ReadOnly):
            print("Failed to open audio play buffer")
            return False
            
        self.audio_play_buffer.seek(0)
        self.playback_start_time = 0
        return True
    
    def get_audio_data(self):
        """Get current audio data"""
        return self.audio_data
    
    def get_sample_rate(self):
        """Get current sample rate"""
        return self.current_sample_rate
