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
    QFrame, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QFont, QCursor, QPixmap, QImage
from PyQt5.QtCore import Qt, QIODevice, QBuffer, QByteArray, QFile, QDataStream, QTimer
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
        fig.tight_layout() # Adjust plot to prevent labels from being cut off

        # --- NEW: Add a progress line ---
        self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
        self.total_time_sec = 0

        # Initialize the FigureCanvas
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_spectrogram(self, samples, sample_rate):
        """Clears the axes and plots a new spectrogram."""
        try:
            self.axes.clear()
            # Re-add the progress line after clearing
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)

            self.axes.specgram(samples, Fs=sample_rate, cmap='viridis')
            self.axes.set_title('Spectrogram')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Frequency (Hz)')
            
            # Set the x-axis limit to the total time
            self.total_time_sec = len(samples) / sample_rate
            self.axes.set_xlim(0, self.total_time_sec)
            
            self.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, f'Error: {e}', color='red', ha='center', va='center')
            self.draw()

    def update_progress_line(self, time_sec):
        """Moves the progress line to the specified time."""
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

        # --- PDF State ---
        self.pdf_files = {}  
        self.current_pdf_doc = None
        self.current_pdf_page = 0
        
        # --- UI Initialization ---
        self.init_audio_input()
        self.init_audio_output()
        self.initUI() 
        
        # --- Timer Connection ---
        self.playback_timer.timeout.connect(self.update_playback_progress)
        self.playback_timer.setInterval(50) # 20 updates per second

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
        # (Stylesheet is unchanged)
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

            /* --- NEW: Status Label Style --- */
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
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
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

            /* --- NEW: Audio Control Buttons --- */
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
        # (This function is unchanged)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.main_page.setLayout(layout)
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("Stutter Detection System")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(10)
        self.main_area = QLabel("Select a PDF from the right, or start a new session.")
        self.main_area.setObjectName("mainArea")
        self.main_area.setAlignment(Qt.AlignCenter)
        self.main_area.setWordWrap(True)
        self.main_area.setMinimumSize(400, 400)
        left_panel_layout.addWidget(self.main_area, 1)
        self.pdf_nav_widget = QWidget()
        pdf_nav_layout = QHBoxLayout(self.pdf_nav_widget)
        pdf_nav_layout.setContentsMargins(0, 0, 0, 0)
        self.prev_page_btn = QPushButton("< Prev")
        self.prev_page_btn.setObjectName("prevPageButton")
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.next_page_btn = QPushButton("Next >")
        self.next_page_btn.setObjectName("nextPageButton")
        pdf_nav_layout.addStretch()
        pdf_nav_layout.addWidget(self.prev_page_btn)
        pdf_nav_layout.addWidget(self.page_label)
        pdf_nav_layout.addWidget(self.next_page_btn)
        pdf_nav_layout.addStretch()
        left_panel_layout.addWidget(self.pdf_nav_widget)
        self.pdf_nav_widget.hide()
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(10)
        self.upload_pdf_btn = QPushButton("Upload PDFs")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.pdf_list_widget = QListWidget()
        right_panel_layout.addWidget(self.upload_pdf_btn)
        right_panel_layout.addWidget(self.status_label)
        right_panel_layout.addWidget(self.pdf_list_widget, 1)
        content_layout.addWidget(left_panel_widget, 75)
        content_layout.addWidget(right_panel_widget, 25)
        layout.addWidget(content_widget, 1)
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMinimumHeight(80)
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
        self.upload_btn.clicked.connect(self.on_upload_audio)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.analysis_btn.clicked.connect(self.on_go_to_analysis)
        self.upload_pdf_btn.clicked.connect(self.on_upload_pdfs)
        self.pdf_list_widget.itemClicked.connect(self.on_pdf_item_clicked)
        self.prev_page_btn.clicked.connect(self.on_prev_page)
        self.next_page_btn.clicked.connect(self.on_next_page)


    def init_analysis_page(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.analysis_page.setLayout(layout)
        
        # --- Title Bar ---
        title_bar = QFrame()
        # ... (same as before) ...
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
        
        # Matplotlib canvas
        self.analysis_canvas = PlotCanvas(analysis_plot_container)
        plot_layout.addWidget(self.analysis_canvas, 1) # Add with stretch

        # --- NEW: Audio Controls Widget ---
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
        # ... (same as before) ...
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
        # ... (same as before) ...
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
        
        self.analysis_canvas.clear_plot()

    # --- Resize Event ---
    def resizeEvent(self, event):
        # (This function is unchanged)
        super().resizeEvent(event)
        if self.current_pdf_doc and self.stack.currentIndex() == 0:
            self.render_pdf_page()

    # --- PDF Slot Functions ---
    def on_upload_pdfs(self):
        # (This function is unchanged)
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)")
        for file_path in file_paths:
            if file_path:
                file_name = file_path.split('/')[-1]
                if file_name not in self.pdf_files:
                    self.pdf_files[file_name] = file_path
                    self.pdf_list_widget.addItem(QListWidgetItem(file_name))

    def on_pdf_item_clicked(self, item):
        # (This function is unchanged)
        file_name = item.text()
        file_path = self.pdf_files[file_name]
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
        try:
            self.current_pdf_doc = fitz.open(file_path)
            self.current_pdf_page = 0
            self.render_pdf_page()
            self.pdf_nav_widget.show()
            self.status_label.setText(f"Viewing: {file_name}")
        except Exception as e:
            self.main_area.clear()
            self.main_area.setText("Error opening PDF.")
            self.main_area.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error opening PDF: {e}")
            self.pdf_nav_widget.hide()
            self.current_pdf_doc = None

    def on_prev_page(self):
        # (This function is unchanged)
        if self.current_pdf_doc and self.current_pdf_page > 0:
            self.current_pdf_page -= 1
            self.render_pdf_page()

    def on_next_page(self):
        # (This function is unchanged)
        if self.current_pdf_doc and self.current_pdf_page < self.current_pdf_doc.page_count - 1:
            self.current_pdf_page += 1
            self.render_pdf_page()

    def render_pdf_page(self):
        # (This function is unchanged)
        if not self.current_pdf_doc:
            return
        try:
            page = self.current_pdf_doc.load_page(self.current_pdf_page)
            label_width = self.main_area.width() * 0.98
            label_height = self.main_area.height() * 0.98
            if label_width <= 0 or label_height <= 0: return
            page_rect = page.rect
            zoom_x = label_width / page_rect.width
            zoom_y = label_height / page_rect.height
            zoom = min(zoom_x, zoom_y)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_format = QImage.Format_RGB888
            q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, image_format)
            q_pixmap = QPixmap.fromImage(q_image)
            self.main_area.setPixmap(q_pixmap)
            self.main_area.setAlignment(Qt.AlignCenter)
            self.page_label.setText(f"Page {self.current_pdf_page + 1} / {self.current_pdf_doc.page_count}")
            self.prev_page_btn.setEnabled(self.current_pdf_page > 0)
            self.next_page_btn.setEnabled(self.current_pdf_page < self.current_pdf_doc.page_count - 1)
        except Exception as e:
            self.main_area.clear()
            self.main_area.setText("Error rendering PDF page.")
            self.main_area.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error rendering page: {e}")
            print(f"Error rendering PDF: {e}")

    def clear_pdf_view(self):
        # (This function is unchanged)
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
            self.current_pdf_doc = None
        self.pdf_nav_widget.hide()
        self.main_area.clear()
        self.main_area.setText("Select a PDF, or start a new session.")
        self.main_area.setAlignment(Qt.AlignCenter)
        self.pdf_list_widget.clearSelection()
        self.status_label.setText("Ready")


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
            self.audio_data = self.audio_buffer.data()
            self.current_sample_rate = self.audio_input.format().sampleRate()
            print(f"Recording stopped. {len(self.audio_data)} bytes captured at {self.current_sample_rate} Hz.")
            self.save_recorded_audio() 
            self.audio_buffer.setData(QByteArray()) 
            self.is_recording = False
            self.record_btn.setText("Start")
            self.record_btn.setObjectName("startButton")
            self.record_btn.setStyleSheet("") 
            self.upload_btn.setEnabled(True)
            self.upload_pdf_btn.setEnabled(True)
        else:
            # --- Start Recording ---
            if not self.audio_buffer.open(QIODevice.WriteOnly):
                self.status_label.setText("Failed to open audio buffer for writing.")
                return
            self.audio_data = QByteArray()
            self.current_sample_rate = 0
            self.is_recording = True
            self.audio_input.start(self.audio_buffer)
            self.status_label.setText("Recording... Press 'Stop' to finish.")
            print("Starting recording...")
            self.record_btn.setText("Stop")
            self.record_btn.setObjectName("stopButton")
            self.record_btn.setStyleSheet("") 
            self.upload_btn.setEnabled(False)
            self.upload_pdf_btn.setEnabled(False)

        
    def save_recorded_audio(self):
        # (This function is unchanged)
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
            self.status_label.setText(f"Recording finished. (Not saved)")

    def write_wav_file(self, file_device, pcm_data, audio_format):
        # (This function is unchanged)
        data_len = len(pcm_data)
        stream = QDataStream(file_device)
        stream.setByteOrder(QDataStream.LittleEndian)
        stream.writeRawData(b'RIFF')
        stream.writeInt32(36 + data_len)
        stream.writeRawData(b'WAVE')
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
        stream.writeRawData(b'data')
        stream.writeInt32(data_len)
        file_device.write(pcm_data)
        
    def on_upload_audio(self):
        """Opens a file dialog to load a WAV audio file and parses it."""
        # (This function is unchanged)
        file_path, _ = QFileDialog.getOpenFileNames(self, "Select Audio File", "", "WAV Files (*.wav)")
        if file_path and file_path[0]:
            file_path = file_path[0]
            self.clear_pdf_view() 
            file = QFile(file_path)
            if not file.open(QIODevice.ReadOnly):
                self.status_label.setText(f"Error: Could not open file {file_path}")
                return
            raw_file_data = file.readAll()
            file.close()
            try:
                with io.BytesIO(raw_file_data.data()) as wav_bytes:
                    with wave.open(wav_bytes, 'rb') as wav_file:
                        n_channels = wav_file.getnchannels()
                        samp_width = wav_file.getsampwidth()
                        self.current_sample_rate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        pcm_data = wav_file.readframes(n_frames)
                        if samp_width != 2:
                            raise ValueError(f"Unsupported sample width: {samp_width} (must be 2 bytes / 16-bit)")
                        if n_channels != 1:
                            raise ValueError(f"Unsupported channel count: {n_channels} (must be 1, mono)")
                        self.audio_data = QByteArray(pcm_data)
                        self.main_area.setText("Audio file loaded.\nReady to analyze.")
                        file_name = file_path.split('/')[-1]
                        self.status_label.setText(f"Loaded: {file_name}")
                        print(f"Loaded file: {file_name}, {len(self.audio_data)} bytes PCM data at {self.current_sample_rate} Hz.")
            except Exception as e:
                self.main_area.setText(f"Error: Could not parse WAV file.\nFile must be 16-bit, mono.\n{e}")
                self.status_label.setText(f"Error: Not a valid WAV file.")
                print(f"Error parsing WAV file {file_path}: {e}")
                self.audio_data = QByteArray()
                self.current_sample_rate = 0
            
    def on_go_to_analysis(self):
        """Switches to the analysis view, plots spectrogram, and sets up audio."""
        print("Switching to analysis page...")
        
        if self.audio_output is None:
            # Try to initialize again if it failed the first time
            self.init_audio_output()
            if self.audio_output is None:
                self.analysis_canvas.clear_plot("Error: Audio output device not found.")
                self.stack.setCurrentIndex(1)
                return

        if len(self.audio_data) > 0 and self.current_sample_rate > 0:
            print(f"Analyzing {len(self.audio_data)} bytes of data at {self.current_sample_rate} Hz...")
            
            try:
                # --- Plot Spectrogram ---
                samples = np.frombuffer(self.audio_data, dtype=np.int16)
                self.analysis_canvas.plot_spectrogram(samples, self.current_sample_rate)

                # --- Setup Audio Playback ---
                self.audio_play_buffer.close() # Close if open
                self.audio_play_buffer.setData(self.audio_data)
                self.audio_play_buffer.open(QIODevice.ReadOnly)
                self.audio_play_buffer.seek(0)
                
                # --- TODO: Add your other analysis logic here ---
                # self.stutter_classes_area.setText(results)
                
                # Reset UI
                self.play_pause_btn.setText("▶ Play")
                self.audio_controls_widget.show()
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
            if self.stack.currentIndex() == 0:
                self.status_label.setText("No audio data to analyze.")
            
        self.stack.setCurrentIndex(1)
        
    def on_back_to_main(self):
        """Switches back to main, stops audio, and clears plot."""
        if self.audio_output:
            self.audio_output.stop()
        self.playback_timer.stop()
        self.audio_play_buffer.close()
        
        self.analysis_canvas.clear_plot()
        self.audio_controls_widget.hide()
        self.stack.setCurrentIndex(0)

    # --- NEW Audio Playback Handlers ---

    def on_play_pause_audio(self):
        """Toggles audio playback between Play and Pause."""
        if self.audio_output is None:
            return
            
        state = self.audio_output.state()
        
        if state == QAudio.ActiveState:
            # --- Pause Audio ---
            self.audio_output.suspend()
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
            
        elif state == QAudio.SuspendedState:
            # --- Resume Audio ---
            self.audio_output.resume()
            self.playback_timer.start()
            self.play_pause_btn.setText("❚❚ Pause")
            
        elif state == QAudio.StoppedState or state == QAudio.IdleState:
            # --- Start Audio ---
            self.audio_play_buffer.seek(0) # Always start from beginning
            self.audio_output.start(self.audio_play_buffer)
            self.playback_timer.start()
            self.play_pause_btn.setText("❚❚ Pause")
            self.analysis_canvas.update_progress_line(0)

    def on_seek_audio(self, seconds_to_seek):
        """Seeks the audio forward or backward by a number of seconds."""
        if self.audio_output is None or self.bytes_per_second == 0:
            return
            
        was_playing = self.audio_output.state() == QAudio.ActiveState
        self.audio_output.stop() # Stop playback to seek
        
        # Calculate new position
        current_pos_bytes = self.audio_play_buffer.pos()
        seek_bytes = int(seconds_to_seek * self.bytes_per_second)
        new_pos = current_pos_bytes + seek_bytes
        
        # Clamp position to valid range (0 to size)
        new_pos = max(0, min(new_pos, self.audio_play_buffer.size() - 1))
        
        self.audio_play_buffer.seek(new_pos)
        
        # Manually update the progress line
        self.update_playback_progress()
        
        if was_playing:
            self.audio_output.start(self.audio_play_buffer)

    def update_playback_progress(self):
        """Called by the QTimer to update the plot line."""
        if self.bytes_per_second == 0:
            return
            
        current_pos_bytes = self.audio_play_buffer.pos()
        current_time_sec = current_pos_bytes / self.bytes_per_second
        
        self.analysis_canvas.update_progress_line(current_time_sec)

    def handle_audio_state_change(self, new_state):
        """Resets UI when audio finishes playing."""
        if new_state == QAudio.IdleState:
            # Audio has finished playing
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
            self.audio_play_buffer.seek(0)
            self.analysis_canvas.update_progress_line(0)


# --- Main execution ---
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

