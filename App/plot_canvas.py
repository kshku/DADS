import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# --- Matplotlib Canvas Widget ---
class PlotCanvas(FigureCanvas):
    """A custom widget to embed a Matplotlib plot in a PyQt5 app."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create a new Figure
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.axes = fig.add_subplot(111)
        
        # Style the axes to match the dark theme
        self.axes.set_facecolor('#1e1e1e')
        self.axes.tick_params(axis='x', colors='#e0e0e0')
        self.axes.tick_params(axis='y', colors='#e0e0e0')
        self.axes.xaxis.label.set_color('#e0e0e0')
        self.axes.yaxis.label.set_color('#e0e0e0')
        self.axes.title.set_color('#e0e0e0')
        fig.tight_layout()
         # Adjust plot to prevent labels from being cut off

        # --- Add a progress line ---
        self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
        self.total_time_sec = 0

        # Initialize the FigureCanvas
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_spectrogram(self, samples, sample_rate):
        """Clears the axes and plots a new spectrogram."""
        try:
            self.axes.clear()

            # Ensure samples are a numpy array of floats and normalized if int16
            samples = np.asarray(samples)
            if samples.dtype == np.int16 or samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 32768.0
            else:
                samples = samples.astype(np.float32)

            # Plot spectrogram (use reasonable FFT params to improve appearance)
            self.axes.specgram(samples, Fs=sample_rate, cmap='viridis', NFFT=1024, noverlap=512)
            self.axes.set_title('Spectrogram')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Frequency (Hz)')
            # --- ADD THESE LINES FOR WHITE LABELS ---
            self.axes.set_facecolor('#1e1e1e') # Keep background dark
            self.axes.tick_params(axis='x', colors='#ffffff')
            self.axes.tick_params(axis='y', colors='#ffffff')
            self.axes.xaxis.label.set_color('#ffffff')
            self.axes.yaxis.label.set_color('#ffffff')
            self.axes.title.set_color('#ffffff')
            # --- END OF FIX ---

            # Set the x-axis limit to the total time
            self.total_time_sec = len(samples) / sample_rate
            self.axes.set_xlim(0, self.total_time_sec)

            # Re-add the progress line after plotting so it stays on top
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5, zorder=10)
            
            try:
                self.figure.tight_layout()
            except Exception:
                pass

            self.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, f'Error: {e}', color='red', ha='center', va='center')
            self.draw()

    def plot_waveform(self, samples, sample_rate):
        """Clears the axes and plots a new waveform."""
        try:
            self.axes.clear()

            # Ensure samples are a numpy array of floats and normalize int16
            samples = np.asarray(samples)
            if samples.dtype == np.int16 or samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 32768.0
            else:
                samples = samples.astype(np.float32)

            # Calculate total time and time axis
            self.total_time_sec = len(samples) / sample_rate
            time_axis = np.linspace(0, self.total_time_sec, num=len(samples))

            self.axes.plot(time_axis, samples, color='#3498db', linewidth=0.5)
            self.axes.set_title('Waveform')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Amplitude')
            # --- ADD THESE LINES FOR WHITE LABELS ---
            self.axes.set_facecolor('#1e1e1e') # Keep background dark
            self.axes.tick_params(axis='x', colors='#ffffff')
            self.axes.tick_params(axis='y', colors='#ffffff')
            self.axes.xaxis.label.set_color('#ffffff')
            self.axes.yaxis.label.set_color('#ffffff')
            self.axes.title.set_color('#ffffff')
            # --- END OF FIX ---

            # Set the x-axis limit to the total time
            self.axes.set_xlim(0, self.total_time_sec)
            # Set y-axis limits to just beyond the min/max amplitude
            min_val = np.min(samples)
            max_val = np.max(samples)
            padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            self.axes.set_ylim(min_val - padding, max_val + padding)

            # Re-add the progress line after plotting so it stays on top
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5, zorder=10)
            self.progress_line.set_visible(True)

            # Adjust layout to prevent clipping
            try:
                self.figure.tight_layout()
            except Exception:
                pass

            self.draw()
        except Exception as e:
            print(f"Error plotting waveform: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, f'Error: {e}', color='red', ha='center', va='center')
            self.draw()

    def update_progress_line(self, time_sec):
        """Moves the progress line to the specified time."""
        if time_sec > self.total_time_sec:
             time_sec = self.total_time_sec
        self.progress_line.set_xdata([time_sec, time_sec])
        # Use draw_idle for better performance in animations
        self.draw_idle()

    def clear_plot(self, message="No audio data to analyze"):
        """Clears the axes and displays a text message."""
        self.axes.clear()
        self.total_time_sec = 0
        # Re-add the progress line so it's ready for the next plot
        self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
        self.progress_line.set_visible(False) # Hide it
        
        self.axes.text(0.5, 0.5, message,
                       horizontalalignment='center',
                       verticalalignment='center',
                       color='#e0e0e0',
                       fontsize=12)
        # Clear labels and ticks
        self.axes.set_title('')
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xlim(0, 1) # Reset x-limit
        self.draw()