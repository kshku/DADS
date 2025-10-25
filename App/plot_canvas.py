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
        
        # --- Sliding window parameters ---
        self.window_size = 5.0  # 5 second window
        self.current_window_start = 0.0
        self.all_samples = None
        self.sample_rate = None
        self.current_view_type = None  # 'waveform' or 'spectrogram'

        # Initialize the FigureCanvas
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_spectrogram(self, samples, sample_rate):
        """Stores data and plots the first 5-second window of spectrogram."""
        try:
            # Ensure samples are a numpy array of floats and normalized if int16
            samples = np.asarray(samples)
            if samples.dtype == np.int16 or samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 32768.0
            else:
                samples = samples.astype(np.float32)

            # Store the full audio data
            self.all_samples = samples
            self.sample_rate = sample_rate
            self.total_time_sec = len(samples) / sample_rate
            self.current_view_type = 'spectrogram'
            self.current_window_start = 0.0
            
            # Plot the first 5-second window
            self._plot_window_spectrogram()
            
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, f'Error: {e}', color='red', ha='center', va='center')
            self.draw()
    
    def _plot_window_spectrogram(self):
        """Plot the current 5-second window of spectrogram."""
        try:
            self.axes.clear()
            
            # Calculate window boundaries
            start_time = self.current_window_start
            end_time = min(start_time + self.window_size, self.total_time_sec)
            
            # Extract samples for this window
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            window_samples = self.all_samples[start_sample:end_sample]
            
            # Plot spectrogram
            self.axes.specgram(window_samples, Fs=self.sample_rate, cmap='viridis', NFFT=1024, noverlap=512)
            self.axes.set_title(f'Spectrogram ({start_time:.1f}s - {end_time:.1f}s)')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Frequency (Hz)')
            self.axes.set_facecolor('#1e1e1e')
            self.axes.tick_params(axis='x', colors='#ffffff')
            self.axes.tick_params(axis='y', colors='#ffffff')
            self.axes.xaxis.label.set_color('#ffffff')
            self.axes.yaxis.label.set_color('#ffffff')
            self.axes.title.set_color('#ffffff')
            
            # Set x-axis to show actual time values
            self.axes.set_xlim(0, end_time - start_time)
            
            # Re-add the progress line
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5, zorder=10)
            
            try:
                self.figure.tight_layout()
            except Exception:
                pass
                
            self.draw()
        except Exception as e:
            print(f"Error plotting window spectrogram: {e}")

    def plot_waveform(self, samples, sample_rate):
        """Stores data and plots the first 5-second window of waveform."""
        try:
            # Ensure samples are a numpy array of floats and normalize int16
            samples = np.asarray(samples)
            if samples.dtype == np.int16 or samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 32768.0
            else:
                samples = samples.astype(np.float32)

            # Store the full audio data
            self.all_samples = samples
            self.sample_rate = sample_rate
            self.total_time_sec = len(samples) / sample_rate
            self.current_view_type = 'waveform'
            self.current_window_start = 0.0
            
            # Plot the first 5-second window
            self._plot_window_waveform()
            
        except Exception as e:
            print(f"Error plotting waveform: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, f'Error: {e}', color='red', ha='center', va='center')
            self.draw()
    
    def _plot_window_waveform(self):
        """Plot the current 5-second window of waveform."""
        try:
            self.axes.clear()
            
            # Calculate window boundaries
            start_time = self.current_window_start
            end_time = min(start_time + self.window_size, self.total_time_sec)
            
            # Extract samples for this window
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            window_samples = self.all_samples[start_sample:end_sample]
            
            # Create time axis for this window
            time_axis = np.linspace(0, end_time - start_time, num=len(window_samples))
            
            self.axes.plot(time_axis, window_samples, color='#3498db', linewidth=0.5)
            self.axes.set_title(f'Waveform ({start_time:.1f}s - {end_time:.1f}s)')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Amplitude')
            self.axes.set_facecolor('#1e1e1e')
            self.axes.tick_params(axis='x', colors='#ffffff')
            self.axes.tick_params(axis='y', colors='#ffffff')
            self.axes.xaxis.label.set_color('#ffffff')
            self.axes.yaxis.label.set_color('#ffffff')
            self.axes.title.set_color('#ffffff')
            
            # Set x-axis
            self.axes.set_xlim(0, end_time - start_time)
            
            # Set y-axis limits
            min_val = np.min(window_samples)
            max_val = np.max(window_samples)
            padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            self.axes.set_ylim(min_val - padding, max_val + padding)
            
            # Re-add the progress line
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5, zorder=10)
            self.progress_line.set_visible(True)
            
            try:
                self.figure.tight_layout()
            except Exception:
                pass
                
            self.draw()
        except Exception as e:
            print(f"Error plotting window waveform: {e}")

    def update_progress_line(self, time_sec):
        """Moves the progress line and updates window if needed."""
        if time_sec > self.total_time_sec:
             time_sec = self.total_time_sec
        
        # Calculate which window this time should be in
        target_window_start = int(time_sec / self.window_size) * self.window_size
        
        # Check if we need to change the window (forward or backward)
        if target_window_start != self.current_window_start:
            # We've moved to a different window - replot
            self.current_window_start = target_window_start
            if self.current_view_type == 'spectrogram':
                self._plot_window_spectrogram()
            elif self.current_view_type == 'waveform':
                self._plot_window_waveform()
        
        # Update progress line position relative to current window
        relative_time = time_sec - self.current_window_start
        self.progress_line.set_xdata([relative_time, relative_time])
        # Use draw_idle for better performance in animations
        self.draw_idle()

    def clear_plot(self, message="No audio data to analyze"):
        """Clears the axes and displays a text message."""
        self.axes.clear()
        self.total_time_sec = 0
        self.all_samples = None
        self.sample_rate = None
        self.current_window_start = 0.0
        self.current_view_type = None
        
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