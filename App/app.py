import sys
import fitz  # PyMuPDF, MUST BE INSTALLED: pip install PyMuPDF
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Set the backend for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import wave
import io

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStackedWidget,
    QFrame, QListWidget, QListWidgetItem, QScrollArea,
    QSlider # --- NEW: Added for scrubbing bar ---
)
from PyQt5.QtGui import QFont, QCursor, QPixmap, QImage
from PyQt5.QtCore import (
    Qt, QIODevice, QBuffer, QByteArray, QFile, QDataStream, QTimer
)
from PyQt5.QtMultimedia import (
    QAudioFormat, QAudioInput, QAudioDeviceInfo,
    QAudioOutput, QAudio # Added for playback
)

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

# --- Main Application Window ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- Audio State (Input) ---
        self.audio_input = None
        self.audio_buffer = None
        self.is_recording = False
        
        # --- Audio State (Playback) ---
        self.audio_data = QByteArray() # This holds the raw PCM data
        self.current_sample_rate = 0
        self.audio_format_out = QAudioFormat() # To store playback format
        self.audio_output = None
        self.audio_play_buffer = QBuffer()
        self.playback_timer = QTimer(self) # For syncing plot
        self.bytes_per_second = 0
        self.current_seek_time = 0.0 # *** Tracks seek time ***
        # --- NEW: State for slider drag ---
        self.was_playing_before_drag = False

        # --- Analysis State ---
        self.current_graph_type = 'spectrogram' # default graph
        self.current_samples = None # To store numpy samples

        # --- PDF State ---
        self.pdf_files = {}  
        self.current_pdf_doc = None
        self.current_pdf_page = 0
        self.pdf_zoom = 1.0 # Zoom factor
        self.pdf_fit_to_page = True # Zoom state

        # --- UI Initialization ---
        self.init_audio_input()
        self.init_audio_output()
        self.initUI() 
        
        # --- Timer Connection ---
        self.playback_timer.timeout.connect(self.update_playback_progress)
        self.playback_timer.setInterval(50) # 20 updates per second (50ms)

    def init_audio_input(self):
        """Sets up the QAudioFormat and QAudioInput for recording."""
        format = QAudioFormat()
        format.setSampleRate(44100)
        format.setChannelCount(1)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)
        
        # Store this format for playback
        self.audio_format_out = format

        info = QAudioDeviceInfo.defaultInputDevice()
        if not info.isFormatSupported(format):
            print("Audio input format not supported by device.")
            return

        self.audio_input = QAudioInput(info, format, self)
        self.audio_buffer = QBuffer()
        
    def init_audio_output(self):
        """Sets up the QAudioOutput for playback."""
        info = QAudioDeviceInfo.defaultOutputDevice()
        if not info.isFormatSupported(self.audio_format_out):
            print("Audio output format not supported by device.")
            return
            
        self.audio_output = QAudioOutput(info, self.audio_format_out, self)
        self.audio_output.stateChanged.connect(self.handle_audio_state_change)
        
        # Set a larger buffer to handle 1-2+ minutes of audio data reliably
        self.audio_output.setBufferSize(8 * 1024 * 1024)
        
        # Calculate bytes per second for seeking
        self.bytes_per_second = self.audio_format_out.sampleRate() * \
                                self.audio_format_out.channelCount() * \
                                (self.audio_format_out.sampleSize() // 8)

    def initUI(self):
        # --- Window Properties ---
        self.setWindowTitle("DADS - Stutter Detection System")
        self.setGeometry(100, 100, 1000, 600)
        self.setFont(QFont("SansSerif", 10))

        # --- Global Stylesheet ---
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: SansSerif;
            }
            QPushButton {
                background-color: #34495e;
                color: #ffffff;
                border: none;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                min-width: 110px;
            }
            QPushButton:hover {
                background-color: #4a6580;
            }
            QPushButton:pressed {
                background-color: #2c3e50;
            }
            
            /* Main Content Areas */
            QLabel#mainArea {
                background-color: #1e1e1e;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #333333;
                font-size: 16px;
            }

            /* --- PDF Scroll Area Style --- */
            QScrollArea#pdfScrollArea {
                background-color: #1e1e1e;
                border-radius: 8px;
                border: 1px solid #333333;
            }
            /* Style for the QLabel *inside* the scroll area */
            QScrollArea#pdfScrollArea QLabel#pdfLabel {
                background-color: #1e1e1e;
                padding: 0px; /* Padding is handled by scroll area */
                border: none; /* Border is handled by scroll area */
            }
            
            /* PDF List Styles */
            QListWidget {
                background-color: #1e1e1e;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #333333;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                color: #e0e0e0;
            }
            QListWidget::item:hover {
                background-color: #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #34495e;
                color: white;
            }

            /* --- Status Label Style --- */
            QLabel#statusLabel {
                font-size: 14px;
                font-weight: bold;
                color: #f39c12; /* Amber color */
                padding: 8px;
                border: 1px solid #333;
                border-radius: 6px;
                background-color: #222;
                min-height: 40px;
                qproperty-alignment: 'AlignCenter';
            }
            
            /* Title Bar */
            QFrame#titleBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333333;
                min-height: 40px;
                max-height: 40px;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: bold;
                color: white;
            }

            /* Bottom Control Bar */
            QFrame#bottomBar {
                background-color: #1e1e1e;
                border-top: 1px solid #333333;
                padding: 10px 20px;
            }
            
            /* Specific Button Styles */
            QPushButton#startButton {
                background-color: #27ae60; /* Green */
            }
            QPushButton#startButton:hover {
                background-color: #2ecc71;
            }
            QPushButton#stopButton {
                background-color: #c0392b; /* Red */
            }
            QPushButton#stopButton:hover {
                background-color: #e74c3c;
            }
            
            /* PDF Page Nav Buttons */
            QPushButton#prevPageButton, QPushButton#nextPageButton {
                min-width: 80px;
                padding: 8px 12px;
                font-size: 13px;
            }

            /* --- Zoom Button Styles --- */
            QPushButton#zoomButton {
                font-size: 16px;
                font-weight: bold;
                min-width: 45px;
                max-width: 45px;
                padding: 8px 12px;
                background-color: #2c3e50;
            }
            QPushButton#zoomButton:hover {
                background-color: #34495e;
            }
            QPushButton#zoomFitButton {
                font-size: 13px;
                min-width: 80px;
                padding: 8px 12px;
                background-color: #2c3e50;
            }
            QPushButton#zoomFitButton:hover {
                background-color: #34495e;
            }

            /* --- Audio Control Buttons --- */
            QPushButton#playPauseButton {
                font-size: 16px;
                font-weight: bold;
                min-width: 130px;
            }
            QPushButton#seekButton {
                font-size: 14px;
                font-weight: bold;
                min-width: 80px;
                background-color: #2c3e50;
            }
            QPushButton#seekButton:hover {
                background-color: #34495e;
            }

            /* --- Graph Toggle Buttons --- */
            QPushButton#graphButton {
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
                padding: 8px 12px;
                background-color: #2c3e50;
            }
            QPushButton#graphButton:hover {
                background-color: #34495e;
            }
            QPushButton#graphButton[selected="true"] {
                background-color: #3498db; /* Blue for selected */
                color: white;
            }

            /* --- Scroll Bar Styles --- */
            QScrollBar:vertical {
                border: none;
                background: #2a2a2a; /* Scroll track background */
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #e0e0e0; /* White handle */
                min-height: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #ffffff; /* Brighter white on hover */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            
            QScrollBar:horizontal {
                border: none;
                background: #2a2a2a;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #e0e0e0; /* White handle */
                min-width: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #ffffff; /* Brighter white on hover */
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            
            /* --- NEW: Audio Slider (Scrubbing Bar) Style --- */
            QSlider#audioSlider::groove:horizontal {
                background: #2a2a2a;
                border: 1px solid #2a2a2a;
                height: 8px;
                border-radius: 4px;
            }
            QSlider#audioSlider::handle:horizontal {
                background: #e0e0e0;
                border: 1px solid #c0c0c0;
                width: 16px;
                margin: -4px 0; /* Vertically center handle */
                border-radius: 8px;
            }
            QSlider#audioSlider::handle:horizontal:hover {
                background: #ffffff;
            }
            QSlider#audioSlider::sub-page:horizontal {
                background: #3498db; /* Blue for played part */
                border-radius: 4px;
            }
            QSlider#audioSlider::add-page:horizontal {
                background: #2a2a2a;
                border-radius: 4px;
            }
        """)

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # --- Stacked Widget ---
        self.stack = QStackedWidget()
        
        # --- Page 1: Main View ---
        self.main_page = QWidget()
        self.init_main_page()
        self.stack.addWidget(self.main_page)
        
        # --- Page 2: Analysis View ---
        self.analysis_page = QWidget()
        self.init_analysis_page()
        self.stack.addWidget(self.analysis_page)

        # Add stack to main layout
        main_layout.addWidget(self.stack)

        # --- Post-UI Audio Check ---
        if self.audio_input is None:
            self.status_label.setText("Audio input device not supported.")
            self.record_btn.setEnabled(False)
        if self.audio_output is None:
            self.status_label.setText("Audio output device not supported.")
            # We might want to disable analysis, but for now just log

    def init_main_page(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.main_page.setLayout(layout)
        
        # --- Title Bar ---
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("Stutter Detection System")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)

        # --- Split Content Area ---
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        # Left Panel (PDF View + Nav)
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(10)

        self.pdf_scroll_area = QScrollArea(self) # Use the standard class
        self.pdf_scroll_area.setObjectName("pdfScrollArea")
        self.pdf_scroll_area.setWidgetResizable(True) # Key for fit-to-page
        self.pdf_scroll_area.setAlignment(Qt.AlignCenter) # Center if page is smaller
        self.pdf_scroll_area.setMinimumSize(400, 400) # Give it a good minimum size

        self.pdf_label = QLabel("Select a PDF from the right, or start a new session.")
        self.pdf_label.setObjectName("pdfLabel") # Use new style
        self.pdf_label.setStyleSheet("background-color: #1e1e1e; border: none; padding: 0;") 
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_label.setWordWrap(True)
        
        self.pdf_scroll_area.setWidget(self.pdf_label) # Put the label inside
        
        left_panel_layout.addWidget(self.pdf_scroll_area, 1) # Add scroll area to layout

        # PDF Nav Buttons (with Zoom controls)
        self.pdf_nav_widget = QWidget()
        pdf_nav_layout = QHBoxLayout(self.pdf_nav_widget)
        pdf_nav_layout.setContentsMargins(0, 0, 0, 0)
        pdf_nav_layout.setSpacing(10)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setObjectName("zoomButton")
        self.zoom_reset_btn = QPushButton("Fit Page")
        self.zoom_reset_btn.setObjectName("zoomFitButton")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setObjectName("zoomButton")
        
        self.prev_page_btn = QPushButton("< Prev")
        self.prev_page_btn.setObjectName("prevPageButton")
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.next_page_btn = QPushButton("Next >")
        self.next_page_btn.setObjectName("nextPageButton")
        
        pdf_nav_layout.addStretch()
        pdf_nav_layout.addWidget(self.zoom_out_btn) 
        pdf_nav_layout.addWidget(self.zoom_reset_btn)
        pdf_nav_layout.addWidget(self.zoom_in_btn) 
        pdf_nav_layout.addSpacing(30) # Spacer
        pdf_nav_layout.addWidget(self.prev_page_btn)
        pdf_nav_layout.addWidget(self.page_label)
        pdf_nav_layout.addWidget(self.next_page_btn)
        pdf_nav_layout.addStretch()
        
        left_panel_layout.addWidget(self.pdf_nav_widget)
        self.pdf_nav_widget.hide() # Hide by default

        # Right Panel (PDF List + Status)
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(10)

        self.upload_pdf_btn = QPushButton("Upload PDFs")
        self.status_label = QLabel("Ready") # Status box
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        
        self.pdf_list_widget = QListWidget() # PDF list

        # --- MODIFIED: Layout order changed per user request ---
        right_panel_layout.addWidget(self.status_label)
        right_panel_layout.addWidget(self.pdf_list_widget, 1) # Add with stretch
        right_panel_layout.addWidget(self.upload_pdf_btn)
        # --- End Modification ---

        # Add panels to content layout
        content_layout.addWidget(left_panel_widget, 75)
        content_layout.addWidget(right_panel_widget, 25)
        layout.addWidget(content_widget, 1) # Add with stretch

        # --- Bottom Bar ---
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMinimumHeight(80) # Give it a fixed height
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setSpacing(15)

        self.upload_btn = QPushButton("📁 Upload Audio")
        self.record_btn = QPushButton("Start")
        self.record_btn.setObjectName("startButton")
        self.analysis_btn = QPushButton("🔍 Go to Analysis")

        bottom_layout.addStretch()
        for btn in [self.upload_btn, self.record_btn, self.analysis_btn]:
            bottom_layout.addWidget(btn)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
        bottom_layout.addStretch()

        layout.addWidget(bottom_bar)

        # --- Connections ---
        self.upload_btn.clicked.connect(self.on_upload_audio)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.analysis_btn.clicked.connect(self.on_go_to_analysis)
        self.upload_pdf_btn.clicked.connect(self.on_upload_pdfs)
        self.pdf_list_widget.itemClicked.connect(self.on_pdf_item_clicked)
        self.prev_page_btn.clicked.connect(self.on_prev_page)
        self.next_page_btn.clicked.connect(self.on_next_page)
        
        self.zoom_in_btn.clicked.connect(lambda: self.on_zoom(1.25)) 
        self.zoom_out_btn.clicked.connect(lambda: self.on_zoom(0.8)) 
        self.zoom_reset_btn.clicked.connect(self.on_zoom_fit)


    def init_analysis_page(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.analysis_page.setLayout(layout)
        
        # --- Title Bar ---
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("Analysis View")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)

        # --- Content Area ---
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        # --- Plot Container (now with audio controls) ---
        analysis_plot_container = QWidget()
        analysis_plot_container.setStyleSheet(
            "background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333333; padding: 5px;"
        )
        plot_layout = QVBoxLayout(analysis_plot_container)
        plot_layout.setContentsMargins(5, 5, 5, 5) # Add some padding
        plot_layout.setSpacing(10)
        
        # --- Graph Controls Widget ---
        self.graph_controls_widget = QWidget()
        graph_controls_layout = QHBoxLayout(self.graph_controls_widget)
        graph_controls_layout.setContentsMargins(0, 0, 0, 0)
        graph_controls_layout.setSpacing(10)
        
        self.spec_btn = QPushButton("Spectrogram")
        self.spec_btn.setObjectName("graphButton")
        self.wave_btn = QPushButton("Waveform")
        self.wave_btn.setObjectName("graphButton")
        
        graph_controls_layout.addWidget(self.spec_btn)
        graph_controls_layout.addWidget(self.wave_btn)
        graph_controls_layout.addStretch()
        
        plot_layout.addWidget(self.graph_controls_widget) # Add controls *above* plot
        self.graph_controls_widget.hide() # Hide by default

        # Matplotlib canvas
        self.analysis_canvas = PlotCanvas(analysis_plot_container)
        plot_layout.addWidget(self.analysis_canvas, 1) # Add with stretch
        
        # --- NEW: Audio Scrubbing Bar (Slider) ---
        self.audio_slider = QSlider(Qt.Horizontal)
        self.audio_slider.setObjectName("audioSlider")
        self.audio_slider.setCursor(QCursor(Qt.PointingHandCursor))
        plot_layout.addWidget(self.audio_slider)
        self.audio_slider.hide() # Hide by default

        # --- Audio Controls Widget ---
        self.audio_controls_widget = QWidget()
        audio_controls_layout = QHBoxLayout(self.audio_controls_widget)
        audio_controls_layout.setContentsMargins(0, 0, 0, 0)
        audio_controls_layout.setSpacing(15)

        self.back_btn_5s = QPushButton("<< 5s")
        self.back_btn_5s.setObjectName("seekButton")
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setObjectName("playPauseButton")
        self.fwd_btn_5s = QPushButton("5s >>")
        self.fwd_btn_5s.setObjectName("seekButton")

        audio_controls_layout.addStretch()
        audio_controls_layout.addWidget(self.back_btn_5s)
        audio_controls_layout.addWidget(self.play_pause_btn)
        audio_controls_layout.addWidget(self.fwd_btn_5s)
        audio_controls_layout.addStretch()
        
        plot_layout.addWidget(self.audio_controls_widget) # Add controls below plot
        self.audio_controls_widget.hide() # Hide by default

        # Right-side panel
        self.stutter_classes_area = QLabel("Stutter Classes Detected:\n- None")
        self.stutter_classes_area.setObjectName("stutterListArea")
        self.stutter_classes_area.setStyleSheet(
            "background-color: #1e1e1e; border-radius: 8px; padding: 20px; border: 1px solid #333333; font-size: 16px;"
        )
        self.stutter_classes_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        content_layout.addWidget(analysis_plot_container, 75)
        content_layout.addWidget(self.stutter_classes_area, 25)
        layout.addWidget(content_widget, 1)

        # --- Bottom Bar ---
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMinimumHeight(80)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setSpacing(15)
        self.back_btn = QPushButton("← Back to Main")
        self.back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        bottom_layout.addWidget(self.back_btn)
        bottom_layout.addStretch()
        layout.addWidget(bottom_bar)
        
        # --- Connections ---
        self.back_btn.clicked.connect(self.on_back_to_main)
        self.play_pause_btn.clicked.connect(self.on_play_pause_audio)
        self.back_btn_5s.clicked.connect(lambda: self.on_seek_audio(-5))
        self.fwd_btn_5s.clicked.connect(lambda: self.on_seek_audio(5))
        self.spec_btn.clicked.connect(lambda: self.on_graph_type_changed('spectrogram'))
        self.wave_btn.clicked.connect(lambda: self.on_graph_type_changed('waveform'))
        
        # --- NEW: Slider Connections ---
        self.audio_slider.sliderPressed.connect(self.on_slider_pressed)
        self.audio_slider.sliderMoved.connect(self.on_slider_moved)
        self.audio_slider.sliderReleased.connect(self.on_slider_released)

        self.analysis_canvas.clear_plot()

    # --- Resize Event ---
    def resizeEvent(self, event):
        # We need to render the PDF page again on resize *if* in fit-to-page mode
        super().resizeEvent(event)
        if self.current_pdf_doc and self.stack.currentIndex() == 0 and self.pdf_fit_to_page:
            self.render_pdf_page()

    # --- PDF Slot Functions ---
    def on_upload_pdfs(self):
        """Opens a file dialog to select multiple PDF files."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)")
        for file_path in file_paths:
            if file_path:
                file_name = file_path.split('/')[-1]
                if file_name not in self.pdf_files:
                    self.pdf_files[file_name] = file_path
                    self.pdf_list_widget.addItem(QListWidgetItem(file_name))

    def on_pdf_item_clicked(self, item):
        """Loads and renders the first page of the selected PDF."""
        file_name = item.text()
        file_path = self.pdf_files[file_name]
        
        # Close any existing document
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
            
        try:
            self.current_pdf_doc = fitz.open(file_path)
            self.current_pdf_page = 0
            self.pdf_fit_to_page = True # Default to fit-to-page
            self.pdf_scroll_area.setWidgetResizable(True) # Ensure this is reset
            self.render_pdf_page()
            self.pdf_nav_widget.show()
            self.status_label.setText(f"Viewing: {file_name}")
        except Exception as e:
            self.pdf_label.clear()
            self.pdf_label.setText("Error opening PDF.")
            self.pdf_label.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error opening PDF: {e}")
            self.pdf_nav_widget.hide()
            self.current_pdf_doc = None

    def on_prev_page(self):
        """Renders the previous PDF page."""
        if self.current_pdf_doc and self.current_pdf_page > 0:
            self.current_pdf_page -= 1
            self.render_pdf_page()

    def on_next_page(self):
        """Renders the next PDF page."""
        if self.current_pdf_doc and self.current_pdf_page < self.current_pdf_doc.page_count - 1:
            self.current_pdf_page += 1
            self.render_pdf_page()

    def render_pdf_page(self):
        """Renders the current PDF page based on zoom/fit state."""
        if not self.current_pdf_doc:
            return

        try:
            page = self.current_pdf_doc.load_page(self.current_pdf_page)
            
            zoom = 1.0 # Default
            
            if self.pdf_fit_to_page:
                # --- Fit to Page Logic ---
                self.pdf_scroll_area.setWidgetResizable(True)
                # Get viewport dimensions, subtract a little for padding/border
                view_width = self.pdf_scroll_area.viewport().width() - 10
                view_height = self.pdf_scroll_area.viewport().height() - 10
                
                if view_width <= 0 or view_height <= 0:
                    return # Window is probably minimized

                page_rect = page.rect
                zoom_x = view_width / page_rect.width
                zoom_y = view_height / page_rect.height
                zoom = min(zoom_x, zoom_y) # Use smallest zoom to fit
                self.pdf_zoom = zoom # Store this as the current "fit" zoom
            
            else:
                # --- Manual Zoom Logic ---
                self.pdf_scroll_area.setWidgetResizable(False) # THIS IS THE KEY
                zoom = self.pdf_zoom
            
            # Create pixmap
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to QPixmap
            image_format = QImage.Format_RGB888
            q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, image_format)
            q_pixmap = QPixmap.fromImage(q_image)

            # Set pixmap to the *inner* label
            self.pdf_label.setPixmap(q_pixmap)
            
            # --- THE FIX: Force the label to resize to the pixmap's size ---
            self.pdf_label.adjustSize() 
            
            # When fitting, center it. When zoomed, this has no effect.
            self.pdf_label.setAlignment(Qt.AlignCenter) 

            # Update page label
            self.page_label.setText(f"Page {self.current_pdf_page + 1} / {self.current_pdf_doc.page_count}")
            
            # Update button states
            self.prev_page_btn.setEnabled(self.current_pdf_page > 0)
            self.next_page_btn.setEnabled(self.current_pdf_page < self.current_pdf_doc.page_count - 1)
            
        except Exception as e:
            self.pdf_label.clear()
            self.pdf_label.setText("Error rendering PDF page.")
            self.pdf_label.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error rendering page: {e}")
            print(f"Error rendering PDF: {e}")

    def clear_pdf_view(self):
        """Clears the PDF view and resets the state."""
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
            self.current_pdf_doc = None
            
        self.pdf_nav_widget.hide()
        self.pdf_label.clear() # Target the inner label
        self.pdf_label.setText("Select a PDF, or start a new session.")
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_list_widget.clearSelection() # Deselect item in list
        self.status_label.setText("Ready")
        
        # Reset zoom state
        self.pdf_fit_to_page = True
        self.pdf_scroll_area.setWidgetResizable(True)
        self.pdf_zoom = 1.0


    # --- Zoom Slot Functions ---
    
    def on_zoom(self, factor):
        """Zooms in or out by a given factor."""
        if not self.current_pdf_doc:
            return
            
        self.pdf_fit_to_page = False # We are now in manual zoom mode
        self.pdf_zoom *= factor
        # Add some min/max zoom limits
        self.pdf_zoom = max(0.1, min(self.pdf_zoom, 8.0)) # 10% to 800%
        
        self.render_pdf_page()

    def on_zoom_fit(self):
        """Resets zoom to fit the page."""
        if not self.current_pdf_doc:
            return
            
        self.pdf_fit_to_page = True # We are now in fit-to-page mode
        self.render_pdf_page()

    # --- Audio Slot Functions ---

    def toggle_recording(self):
        """Starts or stops the audio recording."""
        if self.audio_input is None:
            self.status_label.setText("Audio input device not initialized.")
            return

        if self.is_recording:
            # --- Stop Recording ---
            self.audio_input.stop()
            self.audio_buffer.close()
            
            # Store the data
            self.audio_data = self.audio_buffer.data()
            self.current_sample_rate = self.audio_input.format().sampleRate()
            print(f"Recording stopped. {len(self.audio_data)} bytes captured at {self.current_sample_rate} Hz.")
            
            # Prompt user to save the file
            self.save_recorded_audio() 
            
            # Clear the buffer for the next recording
            self.audio_buffer.setData(QByteArray()) 
            
            # Reset UI
            self.is_recording = False
            self.record_btn.setText("Start")
            self.record_btn.setObjectName("startButton")
            self.record_btn.setStyleSheet("") # Re-apply stylesheet
            self.upload_btn.setEnabled(True)
            self.upload_pdf_btn.setEnabled(True)
        
        else:
            # --- Start Recording ---
            if not self.audio_buffer.open(QIODevice.WriteOnly):
                self.status_label.setText("Failed to open audio buffer for writing.")
                return

            # Clear any old data
            self.audio_data = QByteArray()
            self.current_sample_rate = 0
            
            # Start
            self.is_recording = True
            self.audio_input.start(self.audio_buffer)
            
            # Update UI
            self.status_label.setText("Recording... Press 'Stop' to finish.")
            print("Starting recording...")
            self.record_btn.setText("Stop")
            self.record_btn.setObjectName("stopButton")
            self.record_btn.setStyleSheet("") # Re-apply stylesheet
            self.upload_btn.setEnabled(False) # Disable upload while recording
            self.upload_pdf_btn.setEnabled(False)

        
    def save_recorded_audio(self):
        """Opens a save dialog and writes the recorded audio to a WAV file."""
        if not self.audio_data or len(self.audio_data) == 0:
            print("No audio data to save.")
            self.status_label.setText("No audio data was recorded to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", "", "WAV Files (*.wav)")
        
        if file_path:
            if not file_path.endswith(".wav"):
                file_path += ".wav"
                
            file = QFile(file_path)
            if not file.open(QIODevice.WriteOnly):
                self.status_label.setText(f"Error: Could not open file {file_path}")
                print(f"Error opening file {file_path}")
                return
            
            try:
                # Write the WAV file with a proper header
                self.write_wav_file(file, self.audio_data, self.audio_input.format())
                file_name = file_path.split('/')[-1]
                self.status_label.setText(f"Recording saved:\n{file_name}")
                print(f"Recording saved to {file_path}")
            except Exception as e:
                self.status_label.setText(f"Error saving file: {e}")
                print(f"Error saving file: {e}")
            finally:
                file.close()
        else:
            # User cancelled the save dialog
            self.status_label.setText(f"Recording finished. (Not saved)")

    def write_wav_file(self, file_device, pcm_data, audio_format):
        """Writes a WAV file header and PCM data to a QIODevice."""
        data_len = len(pcm_data)
        
        stream = QDataStream(file_device)
        stream.setByteOrder(QDataStream.LittleEndian)

        # RIFF Header
        stream.writeRawData(b'RIFF')
        stream.writeInt32(36 + data_len) # Total file size - 8
        stream.writeRawData(b'WAVE')

        # Format Chunk
        stream.writeRawData(b'fmt ')
        stream.writeInt32(16) # Subchunk1Size (16 for PCM)
        stream.writeInt16(1)  # AudioFormat (1 for PCM)
        
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
        
        # Write actual data
        file_device.write(pcm_data)
        
    def on_upload_audio(self):
        """Opens a file dialog to load a WAV audio file and parses it."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio File", "", "WAV Files (*.wav)")
        
        if file_paths and file_paths[0]: # Check if list is not empty
            file_path = file_paths[0]
            
            # Clear the PDF view if a file is loaded
            self.clear_pdf_view() 
            
            file = QFile(file_path)
            if not file.open(QIODevice.ReadOnly):
                self.status_label.setText(f"Error: Could not open file {file_path}")
                return
            
            # Read all file data
            raw_file_data = file.readAll()
            file.close()

            # Parse WAV file using Python's 'wave' module
            try:
                with io.BytesIO(raw_file_data.data()) as wav_bytes:
                    with wave.open(wav_bytes, 'rb') as wav_file:
                        n_channels = wav_file.getnchannels()
                        samp_width = wav_file.getsampwidth()
                        self.current_sample_rate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        
                        # Read audio data
                        pcm_data = wav_file.readframes(n_frames)
                        
                        # --- Validation ---
                        if samp_width != 2: # 16-bit
                            raise ValueError(f"Unsupported sample width: {samp_width} (must be 2 bytes / 16-bit)")
                        if n_channels != 1:
                            raise ValueError(f"Unsupported channel count: {n_channels} (must be 1, mono)")
                        
                        # Store data
                        self.audio_data = QByteArray(pcm_data)
                        
                        # Update UI
                        self.pdf_label.setText("Audio file loaded.\nReady to analyze.")
                        self.pdf_label.setAlignment(Qt.AlignCenter)
                        file_name = file_path.split('/')[-1]
                        self.status_label.setText(f"Loaded: {file_name}")
                        print(f"Loaded file: {file_name}, {len(self.audio_data)} bytes PCM data at {self.current_sample_rate} Hz.")
                        
            except Exception as e:
                self.pdf_label.setText(f"Error: Could not parse WAV file.\nFile must be 16-bit, mono.\n{e}")
                self.pdf_label.setAlignment(Qt.AlignCenter)
                self.status_label.setText(f"Error: Not a valid WAV file.")
                print(f"Error parsing WAV file {file_path}: {e}")
                self.audio_data = QByteArray()
                self.current_sample_rate = 0
            
    def on_go_to_analysis(self):
        """Switches to the analysis view, plots spectrogram, and sets up audio."""
        print("Switching to analysis page...")
        
        # Ensure output is initialized
        if self.audio_output is None:
            self.init_audio_output()
            if self.audio_output is None:
                self.analysis_canvas.clear_plot("Error: Audio output device not found.")
                self.stack.setCurrentIndex(1)
                return

        if len(self.audio_data) > 0 and self.current_sample_rate > 0:
            print(f"Analyzing {len(self.audio_data)} bytes of data at {self.current_sample_rate} Hz...")
            
            try:
                # --- Plot Spectrogram ---
                # Convert QByteArray data to numpy array and store it
                self.current_samples = np.frombuffer(self.audio_data, dtype=np.int16)
                
                # --- Set default graph and draw it ---
                self.current_graph_type = 'spectrogram' # Always default to spectrogram
                self.draw_current_graph()

                # --- Setup Audio Playback ---
                self.audio_play_buffer.close() # Close if open
                self.audio_play_buffer.setData(self.audio_data)
                self.audio_play_buffer.open(QIODevice.ReadOnly)
                self.audio_play_buffer.seek(0)
                
                # --- TODO: Add your other analysis logic here ---
                # e.g., run model, get results
                # self.stutter_classes_area.setText(results)
                
                # Reset seek time
                self.current_seek_time = 0.0
                
                # --- NEW: Setup Slider ---
                total_time_ms = int(self.analysis_canvas.total_time_sec * 1000)
                self.audio_slider.setRange(0, total_time_ms)
                self.audio_slider.setValue(0)
                self.audio_slider.show()
                
                # Reset UI
                self.play_pause_btn.setText("▶ Play")
                self.audio_controls_widget.show()
                self.graph_controls_widget.show() # Show graph toggles
                self.analysis_canvas.update_progress_line(0)

            except Exception as e:
                print(f"Error during analysis or plotting: {e}")
                self.analysis_canvas.clear_plot(f"Error: {e}")
                self.status_label.setText("Error during analysis.")

        else:
            # No valid data
            print("No valid audio data to analyze.")
            self.analysis_canvas.clear_plot("No audio data to analyze")
            self.audio_controls_widget.hide()
            self.graph_controls_widget.hide() # Hide graph toggles
            self.audio_slider.hide() # Hide slider
            if self.stack.currentIndex() == 0:
                self.status_label.setText("No audio data to analyze.")
            
        self.stack.setCurrentIndex(1)
        
    def on_back_to_main(self):
        """Switches back to main, stops audio, and clears plot."""
        if self.audio_output:
            self.audio_output.stop()
        self.playback_timer.stop()
        self.audio_play_buffer.close()
        
        # Reset seek time
        self.current_seek_time = 0.0
        
        self.analysis_canvas.clear_plot()
        self.audio_controls_widget.hide()
        self.graph_controls_widget.hide() # Hide graph toggles
        self.audio_slider.hide() # Hide slider
        self.current_samples = None # Clear sample data
        self.stack.setCurrentIndex(0)

    # --- Audio Playback Handlers ---

    def draw_current_graph(self):
        """Draws the currently selected graph type."""
        if self.current_samples is None:
            print("No samples to draw.")
            return

        if self.current_graph_type == 'spectrogram':
            self.analysis_canvas.plot_spectrogram(self.current_samples, self.current_sample_rate)
            self.spec_btn.setProperty('selected', True)
            self.wave_btn.setProperty('selected', False)
        elif self.current_graph_type == 'waveform':
            self.analysis_canvas.plot_waveform(self.current_samples, self.current_sample_rate)
            self.spec_btn.setProperty('selected', False)
            self.wave_btn.setProperty('selected', True)
            
        # Re-polish buttons to update their style based on the 'selected' property
        self.spec_btn.style().unpolish(self.spec_btn)
        self.spec_btn.style().polish(self.spec_btn)
        self.wave_btn.style().unpolish(self.wave_btn)
        self.wave_btn.style().polish(self.wave_btn)

    def on_graph_type_changed(self, graph_type):
        """Switches the graph type and redraws the plot."""
        if graph_type == self.current_graph_type or self.current_samples is None:
            return # Don't redraw if it's the same graph
            
        self.current_graph_type = graph_type
        self.draw_current_graph()
        
        # --- IMPORTANT: Update the progress line to the correct current time ---
        # The new plot reset the line to 0, so we must move it.
        
        current_time_sec = 0.0
        if self.audio_output:
            # This formula correctly gets the time whether playing or paused
            # We add processedUSecs() *only if* it's currently playing
            current_time_sec = self.current_seek_time
            if self.audio_output.state() == QAudio.ActiveState:
                 current_time_sec += (self.audio_output.processedUSecs() / 1_000_000.0)

            # Handle the "just finished" state
            if self.audio_output.state() == QAudio.IdleState:
                current_time_sec = self.analysis_canvas.total_time_sec
        
        self.analysis_canvas.update_progress_line(current_time_sec)


    def on_play_pause_audio(self):
        """Toggles audio playback between Play and Pause."""
        if self.audio_output is None:
            return
            
        state = self.audio_output.state()
        
        if state == QAudio.ActiveState:
            # --- Pause Audio ---
            self.playback_timer.stop()
            
            microseconds_played = self.audio_output.processedUSecs()
            self.current_seek_time += (microseconds_played / 1_000_000.0)
            self.current_seek_time = min(self.current_seek_time, self.analysis_canvas.total_time_sec)
            
            self.audio_output.stop() 
            self.play_pause_btn.setText("▶ Play")
            
            # Update slider to paused position
            self.audio_slider.setValue(int(self.current_seek_time * 1000))
            
        elif state == QAudio.StoppedState or state == QAudio.IdleState or state == QAudio.SuspendedState:
            # --- Play Audio ---
            if state == QAudio.IdleState:
                print("Audio finished, restarting from 0.")
                self.current_seek_time = 0.0
                self.analysis_canvas.update_progress_line(0)
                self.audio_slider.setValue(0) # Reset slider
            
            if state == QAudio.SuspendedState:
                self.audio_output.stop()
            
            # --- Seek to the current_seek_time ---
            # (We call this *after* handling IdleState to ensure it seeks to 0)
            self.seek_to_time(self.current_seek_time, force_play_state=True)
            self.play_pause_btn.setText("❚❚ Pause")


    def on_seek_audio(self, seconds_to_seek):
        """Seeks the audio forward or backward by a number of seconds."""
        if self.audio_output is None or self.bytes_per_second == 0:
            return

        state = self.audio_output.state()
        was_playing = (state == QAudio.ActiveState)
        
        current_time_sec = self.current_seek_time 
        
        if state == QAudio.ActiveState:
            microseconds_played = self.audio_output.processedUSecs()
            current_time_sec += (microseconds_played / 1_000_000.0)
            
        # Calculate new *absolute* time
        new_time_sec = current_time_sec + seconds_to_seek
        
        # Seek and resume previous play state
        self.seek_to_time(new_time_sec, force_play_state=was_playing)

    # --- NEW: Master Seek Function ---
    def seek_to_time(self, new_time_sec, force_play_state=None):
        """
        Seeks audio to a specific time.
        :param new_time_sec: The absolute time in seconds to seek to.
        :param force_play_state: 
            - None: Resumes previous state (if it was playing, play again).
            - True: Forces playback to start.
            - False: Forces playback to be stopped (paused).
        """
        if self.audio_output is None or self.bytes_per_second == 0:
            return

        # Determine if audio *was* playing before this seek
        was_playing = (self.audio_output.state() == QAudio.ActiveState)
        
        # Stop audio and timer to perform the seek
        self.audio_output.stop()
        self.playback_timer.stop()
        
        # Clamp time to be within 0 and total duration
        new_time_sec = max(0, min(new_time_sec, self.analysis_canvas.total_time_sec))

        # Store this as the new *base* time
        self.current_seek_time = new_time_sec 

        # Calculate new byte position from the new *absolute* time
        new_pos_bytes = int(new_time_sec * self.bytes_per_second)
        new_pos_bytes = (new_pos_bytes // 2) * 2 # Align
        
        if not self.audio_play_buffer.seek(new_pos_bytes):
             print(f"Warning: Failed to seek audio buffer to {new_pos_bytes} bytes.")

        # Manually update the progress line and slider to the new seeked time
        self.analysis_canvas.update_progress_line(self.current_seek_time)
        self.audio_slider.setValue(int(self.current_seek_time * 1000))
        
        # Decide whether to play again
        play_now = False
        if force_play_state is None:
            play_now = was_playing # Resume previous state
        elif force_play_state is True:
            play_now = True # Force play
        # if force_play_state is False, play_now remains False (force pause)
            
        if play_now:
            self.audio_output.reset() 
            self.audio_output.start(self.audio_play_buffer)
            self.playback_timer.start()
            self.play_pause_btn.setText("❚❚ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")


    # --- NEW: Slider Slot Functions ---

    def on_slider_pressed(self):
        """Called when user first clicks the slider."""
        if self.audio_output is None: return
        
        self.was_playing_before_drag = (self.audio_output.state() == QAudio.ActiveState)
        
        # Stop playback while dragging
        self.audio_output.stop()
        self.playback_timer.stop()

    def on_slider_moved(self, position_ms):
        """Called when user is dragging the slider."""
        if self.audio_output is None: return
        
        new_time_sec = position_ms / 1000.0
        
        # Store the new base time
        self.current_seek_time = new_time_sec
        
        # Update the plot line *live*
        self.analysis_canvas.update_progress_line(self.current_seek_time)

    def on_slider_released(self):
        """Called when user releases the slider."""
        if self.audio_output is None: return
        
        new_time_sec = self.audio_slider.value() / 1000.0
        
        # Do the actual seek and resume playback if it was playing before
        self.seek_to_time(new_time_sec, force_play_state=self.was_playing_before_drag)


    def update_playback_progress(self):
        """Called by the QTimer to update the plot line and slider."""
        if self.audio_output and self.audio_output.state() == QAudio.ActiveState:
            
            microseconds_played = self.audio_output.processedUSecs()
            current_time_sec = self.current_seek_time + (microseconds_played / 1_000_000.0)
            
            if current_time_sec <= self.analysis_canvas.total_time_sec:
                self.analysis_canvas.update_progress_line(current_time_sec)
                # --- NEW: Update slider position ---
                # Use setBlocking(False) to prevent slider from fighting user input
                self.audio_slider.blockSignals(True) # Don't fire signals
                self.audio_slider.setValue(int(current_time_sec * 1000))
                self.audio_slider.blockSignals(False) # Re-enable signals
            else:
                self.analysis_canvas.update_progress_line(self.analysis_canvas.total_time_sec)


    def handle_audio_state_change(self, new_state):
        """Resets UI when audio finishes playing."""
        if new_state == QAudio.IdleState:
            # Audio has finished playing
            print("Audio state changed to Idle (finished).")
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
            
            # Force-reset the device and our state variables
            self.audio_output.reset() 
            self.current_seek_time = 0.0 # Reset seek time to 0
            
            # Set line and slider to the very end
            if self.analysis_canvas.total_time_sec > 0:
                self.analysis_canvas.update_progress_line(self.analysis_canvas.total_time_sec)
                self.audio_slider.setValue(self.audio_slider.maximum())
            
            self.audio_play_buffer.seek(0)


# --- Main execution ---
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()