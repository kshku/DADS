"""
Main Window Module
Main application window that coordinates all widgets
"""

import glob
import os

from analysis_widget import AnalysisWidget
from audio_handler import AudioHandler
from pdf_viewer_widget import PDFViewerWidget
from PyQt5.QtCore import QFile, QIODevice, Qt
from PyQt5.QtGui import QCursor, QFont
from PyQt5.QtMultimedia import QAudio
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QWidget):
    """Main application window"""

    # PDF folder path - relative to the App directory
    PDF_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Passages")

    # Application stylesheet
    STYLESHEET = """
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
        QLabel#mainArea {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #333333;
            font-size: 16px;
        }
        QScrollArea#pdfScrollArea {
            background-color: #1e1e1e;
            border-radius: 8px;
            border: 1px solid #333333;
        }
        QScrollArea#pdfScrollArea QLabel#pdfLabel {
            background-color: #1e1e1e;
            padding: 0px;
            border: none;
        }
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
        QLabel#statusLabel {
            font-size: 14px;
            font-weight: bold;
            color: #f39c12;
            padding: 8px;
            border: 1px solid #333;
            border-radius: 6px;
            background-color: #222;
            min-height: 40px;
            qproperty-alignment: 'AlignCenter';
        }
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
        QFrame#bottomBar {
            background-color: #1e1e1e;
            border-top: 1px solid #333333;
            padding: 10px 20px;
        }
        QPushButton#startButton {
            background-color: #27ae60;
        }
        QPushButton#startButton:hover {
            background-color: #2ecc71;
        }
        QPushButton#stopButton {
            background-color: #c0392b;
        }
        QPushButton#stopButton:hover {
            background-color: #e74c3c;
        }
        QPushButton#prevPageButton, QPushButton#nextPageButton {
            min-width: 80px;
            padding: 8px 12px;
            font-size: 13px;
        }
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
            background-color: #3498db;
            color: white;
        }
        QPushButton#exportButton {
            background-color: #00695c;
            color: #ffffff;
            border: none;
            padding: 12px 18px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            min-width: 110px;
        }
        QPushButton#exportButton:hover {
            background-color: #00897b;
        }
        QPushButton#exportButton:pressed {
            background-color: #004d40;
        }
        QScrollBar:vertical {
            border: none;
            background: #2a2a2a;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #e0e0e0;
            min-height: 25px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #ffffff;
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
            background: #e0e0e0;
            min-width: 25px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #ffffff;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            background: none;
            border: none;
            width: 0px;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }
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
            margin: -4px 0;
            border-radius: 8px;
        }
        QSlider#audioSlider::handle:horizontal:hover {
            background: #ffffff;
        }
        QSlider#audioSlider::sub-page:horizontal {
            background: #3498db;
            border-radius: 4px;
        }
        QSlider#audioSlider::add-page:horizontal {
            background: #2a2a2a;
            border-radius: 4px;
        }
    """

    def __init__(self):
        super().__init__()

        # Initialize audio handler
        self.audio_handler = AudioHandler(self)
        self.is_recording = False

        self._init_ui()
        self._connect_signals()

        # Load PDFs on startup
        self.load_initial_pdfs()

    def _init_ui(self):
        """Initialize main window UI"""
        self.setWindowTitle("DADS - Stutter Detection System")
        self.setGeometry(100, 100, 1000, 600)
        self.setFont(QFont("SansSerif", 10))
        self.setStyleSheet(self.STYLESHEET)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Stacked widget for pages
        self.stack = QStackedWidget()

        # Main page
        self.main_page = QWidget()
        self._init_main_page()
        self.stack.addWidget(self.main_page)

        # Analysis page
        self.analysis_widget = AnalysisWidget()
        self.analysis_widget.set_audio_handler(self.audio_handler)
        self.stack.addWidget(self.analysis_widget)

        main_layout.addWidget(self.stack)

    def _init_main_page(self):
        """Initialize main page UI"""
        layout = QVBoxLayout(self.main_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title Bar
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("Stutter Detection System")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)

        # Content Area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        # Left Panel - PDF Viewer
        self.pdf_viewer = PDFViewerWidget()

        # Right Panel - Status and Controls
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(10)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        self.pdf_list_widget = QListWidget()

        self.upload_pdf_btn = QPushButton("Upload PDFs")

        right_panel_layout.addWidget(self.status_label)
        right_panel_layout.addWidget(self.pdf_list_widget, 1)
        right_panel_layout.addWidget(self.upload_pdf_btn)

        content_layout.addWidget(self.pdf_viewer, 75)
        content_layout.addWidget(right_panel_widget, 25)
        layout.addWidget(content_widget, 1)

        # Bottom Bar
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

    def _connect_signals(self):
        """Connect all signals and slots"""
        # Main page buttons
        self.upload_btn.clicked.connect(self.on_upload_audio)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.analysis_btn.clicked.connect(self.on_go_to_analysis)

        # PDF viewer
        self.upload_pdf_btn.clicked.connect(self.on_upload_pdfs)
        self.pdf_list_widget.itemClicked.connect(self.pdf_viewer.on_pdf_item_clicked)
        self.pdf_viewer.pdf_status_changed.connect(self.status_label.setText)

        # Analysis widget
        self.analysis_widget.back_requested.connect(self.on_back_to_main)
        self.analysis_widget.back_btn_5s.clicked.connect(lambda: self.on_seek_audio(-5))
        self.analysis_widget.fwd_btn_5s.clicked.connect(lambda: self.on_seek_audio(5))

        # Audio handler state changes
        if self.audio_handler.audio_output:
            self.audio_handler.audio_output.stateChanged.connect(self.handle_audio_state_change)

    def load_initial_pdfs(self):
        """Load all PDF files from the predefined folder"""
        folder_path = self.PDF_FOLDER_PATH

        if not os.path.isdir(folder_path):
            self.status_label.setText(f"Error: Passage folder not found at:\n{folder_path}")
            return

        pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_paths:
            self.status_label.setText("Ready. No PDFs found in the passage folder.")
            return

        for file_path in pdf_paths:
            file_name = os.path.basename(file_path)
            self.pdf_viewer.add_pdf_file(file_name, file_path)
            self.pdf_list_widget.addItem(QListWidgetItem(file_name))

        self.status_label.setText(f"Ready. Loaded {len(pdf_paths)} PDF files.")
        print(f"Loaded {len(pdf_paths)} PDF files from: {folder_path}")

    def on_upload_pdfs(self):
        """Upload PDF files"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)")
        for file_path in file_paths:
            if file_path:
                file_name = file_path.split("/")[-1]
                if file_name not in self.pdf_viewer.pdf_files:
                    self.pdf_viewer.add_pdf_file(file_name, file_path)
                    self.pdf_list_widget.addItem(QListWidgetItem(file_name))

    def toggle_recording(self):
        """Toggle audio recording"""
        if self.is_recording:
            # Stop recording
            audio_data, sample_rate = self.audio_handler.stop_recording()
            print(f"Recording stopped. {len(audio_data)} bytes captured at {sample_rate} Hz.")

            # Save the recorded audio
            self.save_recorded_audio()

            # Reset UI
            self.is_recording = False
            self.record_btn.setText("Start")
            self.record_btn.setObjectName("startButton")
            self.record_btn.setStyleSheet(self.STYLESHEET)
            self.upload_btn.setEnabled(True)
            self.upload_pdf_btn.setEnabled(True)
        else:
            # Start recording
            if self.audio_handler.start_recording():
                self.is_recording = True
                self.status_label.setText("Recording... Press 'Stop' to finish.")
                print("Starting recording...")
                self.record_btn.setText("Stop")
                self.record_btn.setObjectName("stopButton")
                self.record_btn.setStyleSheet(self.STYLESHEET)
                self.upload_btn.setEnabled(False)
                self.upload_pdf_btn.setEnabled(False)
            else:
                self.status_label.setText("Failed to start recording.")

    def save_recorded_audio(self):
        """Save recorded audio to WAV file"""
        if len(self.audio_handler.audio_data) == 0:
            print("No audio data to save.")
            self.status_label.setText("No audio data was recorded to save.")
            return

        # Set default save directory to Recordings folder (relative path)
        recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        default_path = os.path.join(recordings_dir, "recording.wav")

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", default_path, "WAV Files (*.wav)")

        if file_path:
            if not file_path.endswith(".wav"):
                file_path += ".wav"

            file = QFile(file_path)
            if not file.open(QIODevice.WriteOnly):
                self.status_label.setText(f"Error: Could not open file {file_path}")
                print(f"Error opening file {file_path}")
                return

            try:
                self.audio_handler.write_wav_file(
                    file, self.audio_handler.audio_data, self.audio_handler.audio_format_out
                )
                file_name = file_path.split("/")[-1]
                self.status_label.setText(f"Recording saved:\n{file_name}")
                print(f"Recording saved to {file_path}")
            except Exception as e:
                self.status_label.setText(f"Error saving file: {e}")
                print(f"Error saving file: {e}")
            finally:
                file.close()
        else:
            self.status_label.setText("Recording finished. (Not saved)")

    def on_upload_audio(self):
        """Upload audio file"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio File", "", "WAV Files (*.wav)")

        if file_paths and file_paths[0]:
            file_path = file_paths[0]
            self.pdf_viewer.clear_pdf_view()

            if self.audio_handler.load_wav_file(file_path):
                file_name = file_path.split("/")[-1]
                self.status_label.setText(f"Loaded: {file_name}")
                self.pdf_viewer.pdf_label.setText("Audio file loaded.\nReady to analyze.")
                print(f"Loaded file: {file_name}")
            else:
                self.pdf_viewer.pdf_label.setText("Error: Could not parse WAV file.\nFile must be 16-bit, mono.")
                self.status_label.setText("Error: Not a valid WAV file.")

    def on_go_to_analysis(self):
        """Switch to analysis view"""
        print("Switching to analysis page...")
        self.analysis_widget.load_analysis()
        self.stack.setCurrentIndex(1)

    def on_back_to_main(self):
        """Switch back to main view"""
        print("Switching back to main page...")
        self.analysis_widget.unload_analysis()
        self.stack.setCurrentIndex(0)

    def on_play_pause_audio(self):
        """Delegate to AnalysisWidget's play/pause handler"""
        if hasattr(self.analysis_widget, "on_play_pause"):
            self.analysis_widget.on_play_pause()

    def on_seek_audio(self, seconds_to_seek):
        """Seek audio forward or backward"""
        if not self.audio_handler.audio_output or self.audio_handler.bytes_per_second == 0:
            return

        state = self.audio_handler.audio_output.state()
        was_playing = state == QAudio.ActiveState
        current_time_sec = getattr(self.analysis_widget, "current_playback_position_sec", 0.0)
        new_time_sec = current_time_sec + seconds_to_seek
        self.analysis_widget.seek_to_time(new_time_sec, force_play_state=was_playing)

    def handle_audio_state_change(self, new_state):
        """Handle audio state changes"""
        if new_state == QAudio.IdleState:
            print("Audio state changed to Idle (finished).")
            self.analysis_widget.playback_timer.stop()
            self.analysis_widget.play_pause_btn.setText("▶ Play")

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.stack.currentIndex() == 0:
            self.pdf_viewer.render_pdf_page()
