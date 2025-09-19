import tkinter as tk
from tkinter import ttk
from tkinter import DISABLED, NORMAL
import threading
from plots import mel_filterbank, RATE, CHUNK
from audio_utils import AudioRecorder
import matplotlib
matplotlib.use("TkAgg")  # Important for Tkinter compatibility
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import numpy as np
from datetime import datetime
matplotlib.use("TkAgg")  # Important for Tkinter compatibility

class StutterApp:
    def __init__(self, root):
        self.root = root
        self.stop_event = threading.Event()
        self.recorder = AudioRecorder()
        self.recorder.start_stream()  # Start stream on initialization
        self.video_writer = None
        self.recording_video = False
        self.spectrogram_running = False
        
        # Create Recordings directory if it doesn't exist
        if not os.path.exists("Recordings"):
            os.makedirs("Recordings")

        # Main Frame for Center Alignment
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True)

        # Functionalities Row
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(pady=10)

        self.start_btn = tk.Button(self.controls_frame, text="▶ Start Spectrogram", command=self.start_spectrogram)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(self.controls_frame, text="⏹ Stop", command=self.stop_spectrogram, state=DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.record_btn = tk.Button(self.controls_frame, text="🔴 Start Recording", command=self.start_recording)
        self.record_btn.grid(row=0, column=2, padx=5)

        self.stop_record_btn = tk.Button(self.controls_frame, text="💾 Stop Recording", command=self.stop_recording, state=DISABLED)
        self.stop_record_btn.grid(row=0, column=3, padx=5)

        # Stutter Visualization Area
        self.stutter_label = tk.Label(self.main_frame, text="Stutter Visualization", font=("Arial", 14))
        self.stutter_label.pack(pady=10)

        self.stutter_canvas = tk.Canvas(self.main_frame, width=800, height=400, bg="white")
        self.stutter_canvas.pack(pady=5)

        self.status_label = tk.Label(self.main_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def start_spectrogram(self):
        if not self.spectrogram_running:
            self.update_status("Running Spectrogram")
            self.stop_event.clear()
            self.spectrogram_running = True
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=NORMAL)
            threading.Thread(target=self.run_spectrogram_with_video, daemon=True).start()

    def stop_spectrogram(self):
        if self.spectrogram_running:
            self.update_status("Spectrogram Stopped")
            self.stop_event.set()
            self.spectrogram_running = False
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)
            if self.recording_video:
                self.stop_recording()

    def start_recording(self):
        if not self.spectrogram_running:
            self.update_status("Please start the spectrogram first")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_path = os.path.join("Recordings", f"recording_{timestamp}")
        
        try:
            # Ensure Recordings directory exists
            os.makedirs("Recordings", exist_ok=True)
            
            # Start audio recording first
            self.update_status("Recording Audio and Video")
            self.recording_video = True
            self.recorder.start_recording(f"{self.current_recording_path}.wav")
            
            # Set up video writer
            fig = plt.gcf()  # Get current figure
            if fig:
                self.video_writer = FFMpegWriter(
                    fps=30,
                    metadata=dict(title='Spectrogram Recording', artist='StutterApp'),
                    codec='h264',
                    bitrate=-1
                )
                self.video_writer.setup(fig, f"{self.current_recording_path}.mp4", dpi=100)
                
                # Update UI
                self.record_btn.config(state=DISABLED)
                self.stop_record_btn.config(state=NORMAL)
                print(f"🎙 Recording started... Saving to {self.current_recording_path}")
            else:
                print("No active figure found for recording")
                self.update_status("Error: No active figure found")
                self.recording_video = False
                self.recorder.stop_recording()  # Stop audio recording if video fails
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.update_status(f"Error starting recording: {str(e)}")
            self.recording_video = False
            if hasattr(self, 'video_writer') and self.video_writer:
                self.video_writer.finish()
                self.video_writer = None

    def stop_recording(self):
        try:
            if self.recording_video:
                # Stop video recording first
                if hasattr(self, 'video_writer') and self.video_writer:
                    self.video_writer.finish()
                    self.video_writer = None
                
                # Stop audio recording
                self.recorder.stop_recording(f"{self.current_recording_path}.wav")
                self.recording_video = False
                
                # Update UI
                self.update_status("Recording Stopped")
                self.record_btn.config(state=NORMAL)
                self.stop_record_btn.config(state=DISABLED)
                print(f"💾 Saved recording to {self.current_recording_path}")
        except Exception as e:
            print(f"Error stopping recording: {e}")
            self.update_status(f"Error stopping recording: {str(e)}")
        finally:
            # Ensure buttons are reset even if there's an error
            self.recording_video = False
            self.record_btn.config(state=NORMAL)
            self.stop_record_btn.config(state=DISABLED)

    def run_spectrogram_with_video(self):
        try:
            N_FFT = 1024
            HISTORY_LEN = 100
            DPI = 100
            mel_fb = mel_filterbank(n_filters=40, n_fft=N_FFT, sr=RATE)
            mel_spec = np.zeros((mel_fb.shape[0], HISTORY_LEN))

            fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
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
                if self.stop_event.is_set():
                    plt.close(fig)
                    return [im]
                try:
                    raw_data = self.recorder.stream.read(CHUNK, exception_on_overflow=False)
                    if raw_data:
                        data = np.frombuffer(raw_data, dtype=np.int16)
                        
                        # FFT processing
                        fft_data = np.fft.rfft(data, n=N_FFT)
                        power_spectrum = np.abs(fft_data) ** 2
                        
                        # Mel spectrogram processing
                        mel_energy = np.dot(mel_fb, power_spectrum)
                        mel_db = 20 * np.log10(mel_energy + 1e-10)
                        
                        # Update visualization
                        mel_spec = np.roll(mel_spec, -1, axis=1)
                        mel_spec[:, -1] = mel_db
                        im.set_array(mel_spec)
                        im.set_clim(vmin=np.min(mel_spec), vmax=np.max(mel_spec))
                        
                        # Capture frame if recording
                        if self.recording_video and hasattr(self, 'video_writer') and self.video_writer:
                            try:
                                self.video_writer.grab_frame()
                            except Exception as e:
                                print(f"Error grabbing video frame: {e}")
                                self.video_writer = None
                                self.recording_video = False
                                self.root.after(0, self.stop_recording)  # Stop recording on main thread
                            
                except Exception as e:
                    print(f"Error in update: {e}")
                return [im]

            # Initialize animation
            ani = animation.FuncAnimation(
                fig, update, interval=30,
                blit=True, cache_frame_data=False
            )
            plt.show()

        except Exception as e:
            print(f"Error in spectrogram: {e}")
            self.update_status(f"Error: {str(e)}")
        finally:
            if self.recording_video:
                self.stop_recording()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")  # Adjusted window size for better visualization
    app = StutterApp(root)
    root.mainloop()
