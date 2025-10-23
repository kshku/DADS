import sys
import fitz
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import wave
import io
import os 
import shutil 

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStackedWidget,
    QFrame, QListWidget, QListWidgetItem,
    QScrollArea
)
from PyQt5.QtGui import QFont, QCursor, QPixmap, QImage
from PyQt5.QtCore import Qt, QIODevice, QBuffer, QByteArray, QFile, QDataStream, QTimer

from PyQt5.QtMultimedia import (
    QAudioFormat, QAudioInput, QAudioDeviceInfo,
    QAudioOutput, QAudio 
)

# --- Matplotlib Canvas Widget ---
class PlotCanvas(FigureCanvas):
    """A custom widget to embed a Matplotlib plot in a PyQt5 app."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e', 
                     layout='constrained') 
        self.axes = fig.add_subplot(111)
        
        self.axes.set_facecolor('#1e1e1e')
        
        self.axes.tick_params(axis='x', colors='#ffffff')
        self.axes.tick_params(axis='y', colors='#ffffff')
        self.axes.xaxis.label.set_color('#ffffff')
        self.axes.yaxis.label.set_color('#ffffff')
        self.axes.title.set_color('#ffffff')
        
        self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
        self.total_time_sec = 0

        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_spectrogram(self, samples, sample_rate):
        """Clears the axes and plots a new spectrogram."""
        try:
            self.axes.clear()
            
            # Re-apply white colors after clearing
            self.axes.set_facecolor('#1e1e1e')
            self.axes.tick_params(axis='x', colors='#ffffff')
            self.axes.tick_params(axis='y', colors='#ffffff')
            self.axes.xaxis.label.set_color('#ffffff')
            self.axes.yaxis.label.set_color('#ffffff')
            self.axes.title.set_color('#ffffff')
            
            self.axes.specgram(samples, Fs=sample_rate, cmap='viridis')
            self.axes.set_title('Spectrogram')
            self.axes.set_xlabel('Time (s)')
            self.axes.set_ylabel('Frequency (Hz)')
            
            self.total_time_sec = len(samples) / sample_rate
            
            self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
            self.progress_line.set_visible(True) 
            
            self.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            self.axes.clear()
            self.axes.text(0.5, 0.5, 'Plotting Error', color='red', ha='center', va='center')
            self.draw()

    def update_progress_line(self, time_sec):
        """Moves the progress line to the specified time."""
        time_sec = max(0, min(time_sec, self.total_time_sec))
        self.progress_line.set_xdata([time_sec, time_sec])
        self.draw_idle()

    def clear_plot(self, message="No audio data to analyze"):
        """Clears the axes and displays a text message."""
        self.axes.clear()
        self.total_time_sec = 0
        
        self.progress_line = self.axes.axvline(0, color='r', linestyle='--', linewidth=1.5)
        self.progress_line.set_visible(False) 
        
        self.axes.text(0.5, 0.5, message,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='#e0e0e0',
                        fontsize=12)
                        
        self.axes.set_title('')
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xlim(0, 1) 
        self.draw()

# --- Main Application Window ---
class MainWindow(QWidget):
    SCROLL_WINDOW_WIDTH = 5.0 # Seconds
    
    # FIX: Defined PDF_DIRECTORY_PATH as a CLASS ATTRIBUTE with your specific path
    PDF_DIRECTORY_PATH = '/home/chethan/WorksSpace/Git/mini-project/DADS/Passage Folder/' 
    
    def __init__(self):
        super().__init__()
        
        # 🟢 PDF Zoom Level
        self.pdf_zoom = 1.0 
        
        # --- Audio State (Input) ---
        self.audio_input = None
        self.audio_buffer = None
        self.is_recording = False
        
        # --- Audio State (Playback) ---
        self.audio_data = QByteArray() 
        self.current_sample_rate = 0
        self.audio_format_out = QAudioFormat() 
        self.audio_output = None
        self.audio_play_buffer = QBuffer()
        self.bytes_per_second = 0
        self.bytes_processed = 0 # To track position based on notify signal

        # --- PDF State ---
        self.pdf_files = {}  
        self.current_pdf_doc = None
        self.current_pdf_page = 0
        
        # --- UI Initialization & Audio Setup ---
        self.init_audio_input()
        self.init_audio_output() 
        self.initUI() 
        
        # Load existing PDFs now that the class attribute is defined
        self.load_initial_pdfs() 
        
        # --- Post-UI Audio Check (only affects Main View buttons) ---
        if self.audio_input is None:
            self.status_label.setText("Audio input format not supported by device.")
            self.record_btn.setEnabled(False)
            
    def init_audio_input(self):
        """Sets up the QAudioFormat and QAudioInput for recording."""
        format = QAudioFormat()
        format.setSampleRate(44100)
        format.setChannelCount(1)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)
        
        self.audio_format_out = format
        
        self.bytes_per_second = format.sampleRate() * \
                                format.channelCount() * \
                                (format.sampleSize() // 8)

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
        
        # Set buffer size for stability against jitters
        self.audio_output.setBufferSize(self.bytes_per_second) 
        self.audio_output.setNotifyInterval(50) # Notify often for smoother line movement
        
        # CRITICAL: Connect notify signal for accurate position tracking (fixes fast line)
        self.audio_output.notify.connect(self.update_playback_progress) 
        
        # Only connect state change for button text update. 
        # The IdleState handler is removed/simplified to avoid premature stopping.
        self.audio_output.stateChanged.connect(self.handle_button_text_update)
        

    def initUI(self):
        # --- Window Properties ---
        self.setWindowTitle("DADS - Stutter Detection System")
        self.setGeometry(100, 100, 1000, 600)
        self.setFont(QFont("SansSerif", 10))

        # --- Global Stylesheet (omitted for brevity) ---
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #e0e0e0; font-family: SansSerif; }
            QPushButton { background-color: #34495e; color: #ffffff; border: none; padding: 12px 18px; border-radius: 6px; font-size: 14px; font-weight: bold; min-width: 110px; }
            QPushButton:hover { background-color: #4a6580; }
            QPushButton:pressed { background-color: #2c3e50; }
            QLabel#mainArea { background-color: #1e1e1e; border-radius: 8px; padding: 20px; border: 1px solid #333333; font-size: 16px; }
            QListWidget { background-color: #1e1e1e; border-radius: 8px; padding: 10px; border: 1px solid #333333; font-size: 14px; }
            QListWidget::item:selected { background-color: #34495e; color: white; }
            QLabel#statusLabel { font-size: 14px; font-weight: bold; color: #f39c12; padding: 8px; border: 1px solid #333; border-radius: 6px; background-color: #222; min-height: 40px; qproperty-alignment: 'AlignCenter'; }
            QFrame#titleBar { background-color: #1e1e1e; border-bottom: 1px solid #333333; min-height: 40px; max-height: 40px; }
            QLabel#titleLabel { font-size: 18px; font-weight: bold; color: white; }
            QFrame#bottomBar { background-color: #1e1e1e; border-top: 1px solid #333333; padding: 10px 20px; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }
            QPushButton#startButton { background-color: #27ae60; }
            QPushButton#stopButton { background-color: #c0392b; }
            QPushButton#playPauseButton { font-size: 16px; font-weight: bold; min-width: 130px; }
            QPushButton#seekButton { font-size: 14px; font-weight: bold; min-width: 80px; background-color: #2c3e50; }
            QPushButton#zoomPdfButton { min-width: 40px; padding: 8px 12px; font-size: 13px; background-color: #2c3e50; }
        """)

        # --- Main Layout (omitted for brevity) ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # --- Stacked Widget (omitted for brevity) ---
        self.stack = QStackedWidget()
        self.main_page = QWidget()
        self.init_main_page()
        self.stack.addWidget(self.main_page)
        self.analysis_page = QWidget()
        self.init_analysis_page()
        self.stack.addWidget(self.analysis_page)
        main_layout.addWidget(self.stack)

    def init_main_page(self):
        # ... (Layout setup omitted for brevity) ...
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.main_page.setLayout(layout)
        
        title_bar = QFrame(); title_bar.setObjectName("titleBar"); title_layout = QHBoxLayout(title_bar); title_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("Stutter Detection System"); title.setObjectName("titleLabel"); title.setAlignment(Qt.AlignCenter); title_layout.addWidget(title)
        layout.addWidget(title_bar)
        
        content_widget = QWidget(); content_layout = QHBoxLayout(content_widget); content_layout.setContentsMargins(15, 15, 15, 15); content_layout.setSpacing(15)
        
        left_panel_widget = QWidget(); left_panel_layout = QVBoxLayout(left_panel_widget); left_panel_layout.setContentsMargins(0, 0, 0, 0); left_panel_layout.setSpacing(10)
        self.main_area = QLabel("Select a PDF from the right, or start a new session."); self.main_area.setObjectName("mainArea"); self.main_area.setAlignment(Qt.AlignCenter); self.main_area.setWordWrap(True)
        self.pdf_scroll_area = QScrollArea(self); self.pdf_scroll_area.setWidgetResizable(True); self.pdf_scroll_area.setWidget(self.main_area)
        self.pdf_scroll_area.setStyleSheet("QScrollArea { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333333; }")
        left_panel_layout.addWidget(self.pdf_scroll_area, 1)
        
        self.pdf_nav_widget = QWidget(); pdf_nav_layout = QHBoxLayout(self.pdf_nav_widget); pdf_nav_layout.setContentsMargins(0, 0, 0, 0)
        self.zoom_in_pdf_btn = QPushButton("Z+"); self.zoom_in_pdf_btn.setObjectName("zoomPdfButton")
        self.zoom_out_pdf_btn = QPushButton("Z-"); self.zoom_out_pdf_btn.setObjectName("zoomPdfButton")
        self.reset_zoom_pdf_btn = QPushButton("Reset"); self.reset_zoom_pdf_btn.setObjectName("zoomPdfButton")
        pdf_nav_layout.addWidget(self.zoom_out_pdf_btn); pdf_nav_layout.addWidget(self.reset_zoom_pdf_btn); pdf_nav_layout.addWidget(self.zoom_in_pdf_btn); pdf_nav_layout.addSpacing(20)
        self.prev_page_btn = QPushButton("< Prev"); self.prev_page_btn.setObjectName("prevPageButton")
        self.page_label = QLabel("Page 0 / 0"); self.page_label.setAlignment(Qt.AlignCenter)
        self.next_page_btn = QPushButton("Next >"); self.next_page_btn.setObjectName("nextPageButton")
        pdf_nav_layout.addStretch(); pdf_nav_layout.addWidget(self.prev_page_btn); pdf_nav_layout.addWidget(self.page_label); pdf_nav_layout.addWidget(self.next_page_btn); pdf_nav_layout.addStretch()
        left_panel_layout.addWidget(self.pdf_nav_widget); self.pdf_nav_widget.hide()
        
        right_panel_widget = QWidget(); right_panel_layout = QVBoxLayout(right_panel_widget); right_panel_layout.setContentsMargins(0, 0, 0, 0); right_panel_layout.setSpacing(10)
        self.upload_pdf_btn = QPushButton("Upload PDFs")
        self.status_label = QLabel("Ready"); self.status_label.setObjectName("statusLabel"); self.status_label.setWordWrap(True)
        self.pdf_list_widget = QListWidget()
        right_panel_layout.addWidget(self.status_label); right_panel_layout.addWidget(self.pdf_list_widget, 1); right_panel_layout.addWidget(self.upload_pdf_btn)
        
        content_layout.addWidget(left_panel_widget, 75); content_layout.addWidget(right_panel_widget, 25)
        layout.addWidget(content_widget, 1)
        
        bottom_bar = QFrame(); bottom_bar.setObjectName("bottomBar"); bottom_bar.setMinimumHeight(80); bottom_layout = QHBoxLayout(bottom_bar); bottom_layout.setSpacing(15)
        self.upload_btn = QPushButton("📁 Upload Audio"); self.record_btn = QPushButton("Start"); self.record_btn.setObjectName("startButton"); self.analysis_btn = QPushButton("🔍 Go to Analysis")
        bottom_layout.addStretch(); 
        for btn in [self.upload_btn, self.record_btn, self.analysis_btn]: bottom_layout.addWidget(btn); btn.setCursor(QCursor(Qt.PointingHandCursor))
        bottom_layout.addStretch(); layout.addWidget(bottom_bar)
        
        self.upload_btn.clicked.connect(self.on_upload_audio); self.record_btn.clicked.connect(self.toggle_recording); self.analysis_btn.clicked.connect(self.on_go_to_analysis)
        self.upload_pdf_btn.clicked.connect(self.on_upload_pdfs); self.pdf_list_widget.itemClicked.connect(self.on_pdf_item_clicked); self.prev_page_btn.clicked.connect(self.on_prev_page); self.next_page_btn.clicked.connect(self.on_next_page)
        self.zoom_in_pdf_btn.clicked.connect(self.on_zoom_in_pdf); self.zoom_out_pdf_btn.clicked.connect(self.on_zoom_out_pdf); self.reset_zoom_pdf_btn.clicked.connect(self.on_reset_zoom_pdf)


    def init_analysis_page(self):
        # ... (Layout setup omitted for brevity) ...
        layout = QVBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0); self.analysis_page.setLayout(layout)
        
        title_bar = QFrame(); title_bar.setObjectName("titleBar"); title_layout = QHBoxLayout(title_bar); title_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("Analysis View"); title.setObjectName("titleLabel"); title.setAlignment(Qt.AlignCenter); title_layout.addWidget(title)
        layout.addWidget(title_bar)

        content_widget = QWidget(); content_layout = QHBoxLayout(content_widget); content_layout.setContentsMargins(15, 15, 15, 15); content_layout.setSpacing(15)

        analysis_plot_container = QWidget(); analysis_plot_container.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333333; padding: 5px;")
        plot_layout = QVBoxLayout(analysis_plot_container); plot_layout.setContentsMargins(5, 5, 5, 5); plot_layout.setSpacing(10)
        
        self.analysis_canvas = PlotCanvas(analysis_plot_container); plot_layout.addWidget(self.analysis_canvas, 1) 

        self.audio_controls_widget = QWidget(); audio_controls_layout = QHBoxLayout(self.audio_controls_widget); audio_controls_layout.setContentsMargins(0, 0, 0, 0); audio_controls_layout.setSpacing(15)
        
        self.zoom_in_btn = QPushButton("🔎 Zoom In (3s)"); self.zoom_in_btn.setObjectName("seekButton")
        self.zoom_out_btn = QPushButton("🔍 Zoom Out (All)"); self.zoom_out_btn.setObjectName("seekButton")
        audio_controls_layout.addWidget(self.zoom_in_btn); audio_controls_layout.addWidget(self.zoom_out_btn); audio_controls_layout.addSpacing(40)
        
        self.back_btn_5s = QPushButton("<< 5s"); self.back_btn_5s.setObjectName("seekButton")
        self.play_pause_btn = QPushButton("▶ Play"); self.play_pause_btn.setObjectName("playPauseButton")
        self.fwd_btn_5s = QPushButton("5s >>"); self.fwd_btn_5s.setObjectName("seekButton")

        audio_controls_layout.addStretch(); audio_controls_layout.addWidget(self.back_btn_5s); audio_controls_layout.addWidget(self.play_pause_btn); audio_controls_layout.addWidget(self.fwd_btn_5s); audio_controls_layout.addStretch()
        
        plot_layout.addWidget(self.audio_controls_widget); self.audio_controls_widget.hide() 

        self.stutter_classes_area = QLabel("Stutter Classes Detected:\n- None"); self.stutter_classes_area.setObjectName("stutterListArea")
        self.stutter_classes_area.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; padding: 20px; border: 1px solid #333333; font-size: 16px;")
        self.stutter_classes_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        content_layout.addWidget(analysis_plot_container, 75); content_layout.addWidget(self.stutter_classes_area, 25)
        layout.addWidget(content_widget, 1)

        bottom_bar = QFrame(); bottom_bar.setObjectName("bottomBar"); bottom_bar.setMinimumHeight(80); bottom_layout = QHBoxLayout(bottom_bar); bottom_layout.setSpacing(15)
        self.back_btn = QPushButton("← Back to Main"); self.back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        bottom_layout.addWidget(self.back_btn); bottom_layout.addStretch(); layout.addWidget(bottom_bar)
        
        self.back_btn.clicked.connect(self.on_back_to_main); self.play_pause_btn.clicked.connect(self.on_play_pause_audio)
        self.back_btn_5s.clicked.connect(lambda: self.on_seek_audio(-5)); self.fwd_btn_5s.clicked.connect(lambda: self.on_seek_audio(5))
        self.zoom_in_btn.clicked.connect(self.on_zoom_in); self.zoom_out_btn.clicked.connect(self.on_zoom_out)
        self.analysis_canvas.clear_plot()
            
    # --- PDF File Management Functions ---
    def load_initial_pdfs(self):
        pdf_path = MainWindow.PDF_DIRECTORY_PATH 
        if not os.path.exists(pdf_path):
            try: os.makedirs(pdf_path); print(f"Created PDF directory: {pdf_path}")
            except OSError as e: self.status_label.setText(f"Error creating PDF folder: {e}"); return
        self.pdf_list_widget.clear(); self.pdf_files = {}
        try:
            for filename in os.listdir(pdf_path):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(pdf_path, filename)
                    self.pdf_files[filename] = file_path 
                    self.pdf_list_widget.addItem(QListWidgetItem(filename))
            self.status_label.setText(f"Loaded {self.pdf_list_widget.count()} PDFs.")
        except Exception as e:
            self.status_label.setText(f"Error listing PDFs: {e}")
            
    def on_upload_pdfs(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files to Upload", "", "PDF Files (*.pdf)")
        if not file_paths: return
        uploaded_count = 0; pdf_path = MainWindow.PDF_DIRECTORY_PATH
        for source_path in file_paths:
            filename = os.path.basename(source_path); dest_path = os.path.join(pdf_path, filename)
            try:
                shutil.copy2(source_path, dest_path)
                if filename not in self.pdf_files:
                    self.pdf_files[filename] = dest_path
                    self.pdf_list_widget.addItem(QListWidgetItem(filename))
                uploaded_count += 1
            except Exception as e:
                self.status_label.setText(f"Error copying {filename}: {e}"); print(f"Error copying file: {e}")
        self.status_label.setText(f"Uploaded {uploaded_count} file(s) to PDF folder.")
        
    def resizeEvent(self, event): 
        super().resizeEvent(event)
        if self.current_pdf_doc and self.stack.currentIndex() == 0: 
            self.render_pdf_page()
            
    def on_pdf_item_clicked(self, item):
        file_name = item.text(); file_path = self.pdf_files[file_name]
        if self.current_pdf_doc: self.current_pdf_doc.close()
        try:
            self.current_pdf_doc = fitz.open(file_path); self.current_pdf_page = 0
            self.pdf_zoom = 1.0; self.render_pdf_page(); self.pdf_nav_widget.show()
            self.status_label.setText(f"Viewing: {file_name}")
        except Exception as e:
            self.main_area.clear(); self.main_area.setText("Error opening PDF."); self.main_area.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error opening PDF: {e}"); self.pdf_nav_widget.hide(); self.current_pdf_doc = None
            
    def on_zoom_in_pdf(self): 
        if self.current_pdf_doc: self.pdf_zoom = min(4.0, self.pdf_zoom + 0.25); self.render_pdf_page()
        
    def on_zoom_out_pdf(self):
        if self.current_pdf_doc: self.pdf_zoom = max(0.25, self.pdf_zoom - 0.25); self.render_pdf_page()
        
    def on_reset_zoom_pdf(self):
        if self.current_pdf_doc: self.pdf_zoom = 1.0; self.render_pdf_page()
        
    def on_prev_page(self):
        if self.current_pdf_doc and self.current_pdf_page > 0: self.current_pdf_page -= 1; self.render_pdf_page()
        
    def on_next_page(self):
        if self.current_pdf_doc and self.current_pdf_page < self.current_pdf_doc.page_count - 1: self.current_pdf_page += 1; self.render_pdf_page()
        
    def render_pdf_page(self):
        if not self.current_pdf_doc: return
        try:
            page = self.current_pdf_doc.load_page(self.current_pdf_page)
            label_width = self.pdf_scroll_area.viewport().width(); label_height = self.pdf_scroll_area.viewport().height()
            if label_width <= 0 or label_height <= 0: label_width = 800; label_height = 600
            page_rect = page.rect
            fit_zoom_x = label_width / page_rect.width; fit_zoom_y = label_height / page_rect.height; fit_zoom = min(fit_zoom_x, fit_zoom_y)
            final_zoom = fit_zoom * self.pdf_zoom
            mat = fitz.Matrix(final_zoom, final_zoom); pix = page.get_pixmap(matrix=mat, alpha=False)
            image_format = QImage.Format_RGB888; q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, image_format)
            q_pixmap = QPixmap.fromImage(q_image)
            self.main_area.setPixmap(q_pixmap); self.main_area.resize(q_pixmap.size())
            self.main_area.setAlignment(Qt.AlignCenter if self.pdf_zoom == 1.0 else Qt.AlignTop | Qt.AlignLeft)
            self.page_label.setText(f"Page {self.current_pdf_page + 1} / {self.current_pdf_doc.page_count}")
            self.prev_page_btn.setEnabled(self.current_pdf_page > 0); self.next_page_btn.setEnabled(self.current_pdf_page < self.current_pdf_doc.page_count - 1)
        except Exception as e:
            self.main_area.clear(); self.main_area.setText("Error rendering PDF page."); self.main_area.setAlignment(Qt.AlignCenter)
            self.status_label.setText(f"Error rendering page: {e}"); print(f"Error rendering PDF: {e}")
            
    def clear_pdf_view(self):
        if self.current_pdf_doc: self.current_pdf_doc.close(); self.current_pdf_doc = None
        self.pdf_nav_widget.hide(); self.main_area.clear(); self.main_area.setText("Select a PDF, or start a new session.")
        self.main_area.setAlignment(Qt.AlignCenter); self.pdf_list_widget.clearSelection(); self.status_label.setText("Ready")
        
    # --- Audio Slot Functions ---
    def toggle_recording(self):
        if self.audio_input is None: self.status_label.setText("Audio input device not initialized."); return
        if self.is_recording:
            self.audio_input.stop(); self.audio_buffer.close(); self.audio_data = self.audio_buffer.data(); self.current_sample_rate = self.audio_input.format().sampleRate()
            print(f"Recording stopped. {len(self.audio_data)} bytes captured at {self.current_sample_rate} Hz.")
            self.save_recorded_audio(); self.audio_buffer.setData(QByteArray()); self.is_recording = False
            self.record_btn.setText("Start"); self.record_btn.setObjectName("startButton"); self.record_btn.setStyleSheet(""); self.upload_btn.setEnabled(True); self.upload_pdf_btn.setEnabled(True)
        else:
            if not self.audio_buffer.open(QIODevice.WriteOnly): self.status_label.setText("Failed to open audio buffer for writing."); return
            self.audio_data = QByteArray(); self.current_sample_rate = 0; self.is_recording = True
            self.audio_input.start(self.audio_buffer); self.status_label.setText("Recording... Press 'Stop' to finish."); print("Starting recording...")
            self.record_btn.setText("Stop"); self.record_btn.setObjectName("stopButton"); self.record_btn.setStyleSheet(""); self.upload_btn.setEnabled(False); self.upload_pdf_btn.setEnabled(False)
            
    def save_recorded_audio(self):
        if not self.audio_data or len(self.audio_data) == 0: print("No audio data to save."); self.status_label.setText("No audio data was recorded to save."); return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", "", "WAV Files (*.wav)")
        if file_path:
            if not file_path.endswith(".wav"): file_path += ".wav"
            file = QFile(file_path)
            if not file.open(QIODevice.WriteOnly): self.status_label.setText(f"Error: Could not open file {file_path}"); print(f"Error opening file {file_path}"); return
            try:
                self.write_wav_file(file, self.audio_data, self.audio_input.format())
                file_name = file_path.split('/')[-1]; self.status_label.setText(f"Recording saved:\n{file_name}"); print(f"Recording saved to {file_path}")
            except Exception as e:
                self.status_label.setText(f"Error saving file: {e}"); print(f"Error saving file: {e}")
            finally: file.close()
        else: self.status_label.setText(f"Recording finished. (Not saved)")
        
    def write_wav_file(self, file_device, pcm_data, audio_format):
        data_len = len(pcm_data); stream = QDataStream(file_device); stream.setByteOrder(QDataStream.LittleEndian)
        stream.writeRawData(b'RIFF'); stream.writeInt32(36 + data_len); stream.writeRawData(b'WAVE')
        stream.writeRawData(b'fmt '); stream.writeInt32(16); stream.writeInt16(1)
        num_channels = audio_format.channelCount(); stream.writeInt16(num_channels)
        sample_rate = audio_format.sampleRate(); stream.writeInt32(sample_rate)
        bits_per_sample = audio_format.sampleSize(); byte_rate = sample_rate * num_channels * (bits_per_sample // 8); stream.writeInt32(byte_rate)
        block_align = num_channels * (bits_per_sample // 8); stream.writeInt16(block_align); stream.writeInt16(bits_per_sample)
        stream.writeRawData(b'data'); stream.writeInt32(data_len); file_device.write(pcm_data)
        
    def on_upload_audio(self):
        file_path, _ = QFileDialog.getOpenFileNames(self, "Select Audio File", "", "WAV Files (*.wav)")
        if file_path and file_path[0]:
            file_path = file_path[0]; self.clear_pdf_view(); file = QFile(file_path)
            if not file.open(QIODevice.ReadOnly): self.status_label.setText(f"Error: Could not open file {file_path}"); return
            raw_file_data = file.readAll(); file.close()
            try:
                with io.BytesIO(raw_file_data.data()) as wav_bytes:
                    with wave.open(wav_bytes, 'rb') as wav_file:
                        n_channels = wav_file.getnchannels(); samp_width = wav_file.getsampwidth(); self.current_sample_rate = wav_file.getframerate(); n_frames = wav_file.getnframes(); pcm_data = wav_file.readframes(n_frames)
                        if samp_width != 2: raise ValueError(f"Unsupported sample width: {samp_width} (must be 2 bytes / 16-bit)")
                        if n_channels != 1: raise ValueError(f"Unsupported channel count: {n_channels} (must be 1, mono)")
                        self.audio_data = QByteArray(pcm_data)
                        self.main_area.setText("Audio file loaded.\nReady to analyze."); file_name = file_path.split('/')[-1]
                        self.status_label.setText(f"Loaded: {file_name}"); print(f"Loaded file: {file_name}, {len(self.audio_data)} bytes PCM data at {self.current_sample_rate} Hz.")
            except Exception as e:
                self.main_area.setText(f"Error: Could not parse WAV file.\nFile must be 16-bit, mono.\n{e}"); self.status_label.setText(f"Error: Not a valid WAV file.")
                print(f"Error parsing WAV file {file_path}: {e}"); self.audio_data = QByteArray(); self.current_sample_rate = 0
            
    def on_go_to_analysis(self):
        print("Switching to analysis page...")
        self.stutter_classes_area.setText("Running Analysis...") 

        if self.audio_output is None: self.init_audio_output()
        if self.audio_output is None: 
            self.stutter_classes_area.setText("Error: Audio output device not found. Cannot plot or play audio."); self.analysis_canvas.clear_plot("No Audio Device"); self.audio_controls_widget.hide(); self.stack.setCurrentIndex(1); return

        if len(self.audio_data) > 0 and self.current_sample_rate > 0:
            print(f"Analyzing {len(self.audio_data)} bytes of data at {self.current_sample_rate} Hz...")
            try:
                samples = np.frombuffer(self.audio_data, dtype=np.int16)
                self.analysis_canvas.plot_spectrogram(samples, self.current_sample_rate)

                # --- Setup Audio Playback Buffer ---
                self.audio_play_buffer.close() 
                self.audio_play_buffer.setData(self.audio_data)
                self.audio_play_buffer.open(QIODevice.ReadOnly)
                
                self.audio_output.stop() 
                self.audio_output.setVolume(1.0) 
                self.bytes_processed = 0 # Initialize processed bytes counter

                self.play_pause_btn.setText("▶ Play"); self.audio_controls_widget.show()
                
                # FIX: Call the main tracking method with 0 processed bytes to initialize the view.
                self.update_playback_progress(0)
                
                self.stutter_classes_area.setText("Stutter Classes Detected:\n- None") 
                total_time = self.analysis_canvas.total_time_sec; DEFAULT_VIEW_WIDTH = self.SCROLL_WINDOW_WIDTH 
                if total_time > 0.0:
                    end_time = min(total_time, DEFAULT_VIEW_WIDTH); self.analysis_canvas.axes.set_xlim(0, end_time); self.analysis_canvas.draw_idle()
            except Exception as e:
                print(f"Error during analysis or plotting: {e}"); self.stutter_classes_area.setText(f"Error during analysis or plotting: \n{e}"); self.analysis_canvas.clear_plot("Analysis Error"); self.audio_controls_widget.hide()
        else:
            print("No valid audio data to analyze."); self.stutter_classes_area.setText("Warning: No audio data to analyze. Please upload or record first."); self.analysis_canvas.clear_plot("No Audio Data"); self.audio_controls_widget.hide()
            
        self.stack.setCurrentIndex(1)
        
    def on_back_to_main(self):
        if self.audio_output:
            self.audio_output.stop()
        self.audio_play_buffer.close()
        self.analysis_canvas.clear_plot()
        self.audio_controls_widget.hide()
        self.stutter_classes_area.setText("Stutter Classes Detected:\n- None") 
        self.stack.setCurrentIndex(0)

    def on_zoom_in(self):
        """Sets the scrolling window width to a tighter 3-second view."""
        self.SCROLL_WINDOW_WIDTH = 3.0
        self.on_apply_scrolling_view()

    def on_zoom_out(self):
        """Sets the scrolling window width to the initial 5-second view, or shows all."""
        self.SCROLL_WINDOW_WIDTH = 5.0 
        self.on_apply_scrolling_view()

    def on_apply_scrolling_view(self):
        """Helper to set the xlim based on the current playback position and SCROLL_WINDOW_WIDTH."""
        total_time = self.analysis_canvas.total_time_sec
        if total_time == 0.0:
            return

        # Use bytes processed for time calculation
        current_time = self.bytes_processed / self.bytes_per_second

        window_width = self.SCROLL_WINDOW_WIDTH
        
        target_offset = window_width * 0.2
        start_time = current_time - target_offset
        end_time = start_time + window_width
        
        if start_time < 0.0:
            start_time = 0.0
            end_time = min(total_time, window_width)
        elif end_time > total_time:
            end_time = total_time
            start_time = max(0.0, total_time - window_width)

        if window_width < total_time:
            self.analysis_canvas.axes.set_xlim(start_time, end_time)
            self.analysis_canvas.draw_idle()
        else:
            self.analysis_canvas.axes.set_xlim(0, total_time)
            self.analysis_canvas.draw_idle()
            
    def on_play_pause_audio(self):
        """
        Manages Play/Pause/Start functionality.
        """
        if self.audio_output is None or not self.audio_play_buffer.isOpen():
            return
            
        state = self.audio_output.state()
        
        if state == QAudio.ActiveState:
            # Currently Playing -> PAUSE/SUSPEND
            self.audio_output.suspend()
            
        elif state == QAudio.SuspendedState:
            # Currently Paused -> RESUME
            self.audio_output.resume()
            
        else: # Covers QAudio.StoppedState or QAudio.IdleState
            
            # 1. Reset position if finished or explicitly stopped
            if self.audio_play_buffer.pos() >= self.audio_play_buffer.size() or state == QAudio.IdleState:
                self.audio_play_buffer.seek(0)
                self.bytes_processed = 0 # Reset byte counter
                self.analysis_canvas.update_progress_line(0) 
                self.on_apply_scrolling_view() 
                
            # 2. Force Stop just in case, then start.
            self.audio_output.stop() 
            
            # 3. Start playback from the current buffer position
            self.audio_output.start(self.audio_play_buffer)
            
            # Use the buffer's current position (0 or paused position)
            current_buffer_pos = self.audio_play_buffer.pos() 
            self.update_playback_progress(current_buffer_pos) 


    def on_seek_audio(self, seconds_to_seek):
        if self.audio_output is None or self.bytes_per_second == 0 or not self.audio_play_buffer.isOpen():
            return
            
        was_playing = self.audio_output.state() == QAudio.ActiveState or self.audio_output.state() == QAudio.SuspendedState

        if was_playing:
            self.audio_output.stop()
            
        current_pos_bytes = self.audio_play_buffer.pos()
        seek_bytes = int(seconds_to_seek * self.bytes_per_second)
        new_pos = current_pos_bytes + seek_bytes
        
        block_align = self.audio_format_out.channelCount() * (self.audio_format_out.sampleSize() // 8)
        
        new_pos = (new_pos // block_align) * block_align
        new_pos = max(0, min(new_pos, self.audio_play_buffer.size() - block_align))
        
        self.audio_play_buffer.seek(new_pos)
        self.bytes_processed = new_pos # Update byte counter after seek
        
        self.update_playback_progress(self.bytes_processed)
        self.on_apply_scrolling_view() # Update view instantly after seeking

        if was_playing:
            self.audio_output.start(self.audio_play_buffer) 


    def update_playback_progress(self, bytes_processed):
        """
        Updates progress line based on the actual bytes processed by the audio hardware or 
        the buffer position if manually called.
        This is connected to audio_output.notify.
        """
        if self.bytes_per_second == 0:
            return
            
        # Update the byte counter based on the notify signal
        self.bytes_processed = bytes_processed
        
        current_time_sec = self.bytes_processed / self.bytes_per_second
        
        # Update line position
        self.analysis_canvas.update_progress_line(current_time_sec)
        
        # Scroll the window
        self.on_apply_scrolling_view()
        
        # MANUAL STOP CHECK (Since IdleState is unreliable)
        # If the audio output is active and we've processed almost all data, stop it.
        # This forces the pause/reset at the end of the file.
        if self.audio_output.state() == QAudio.ActiveState and self.bytes_processed >= len(self.audio_data) * 0.99:
             print("Audio manually finished/resetting playback to start.")
             
             # Force the output device into QAudio.StoppedState
             self.audio_output.stop() 
             
             # Reset all visual and buffer counters
             self.audio_play_buffer.seek(0)
             self.bytes_processed = 0 
             self.analysis_canvas.update_progress_line(0)
             self.on_apply_scrolling_view() 


    def handle_button_text_update(self, new_state):
        if new_state == QAudio.ActiveState:
            self.play_pause_btn.setText("❚❚ Pause")
        elif new_state == QAudio.SuspendedState:
            self.play_pause_btn.setText("▶ Play")
        elif new_state == QAudio.StoppedState or new_state == QAudio.IdleState:
            self.play_pause_btn.setText("▶ Play")

    def handle_audio_state_change(self, new_state):
        """
        Minimal state handler kept to filter out premature IdleState signals.
        The manual check in update_playback_progress handles the actual reset.
        """
        if new_state == QAudio.IdleState:
             # The hardware signalled Idle. We check if the audio really finished.
             if self.bytes_processed < len(self.audio_data) * 0.99:
                  print(f"Warning: IdleState fired prematurely. Processed {self.bytes_processed} of {len(self.audio_data)} bytes.")
             # The manual check in update_playback_progress will catch the full completion.
             # We do nothing here to prevent the hardware from stopping prematurely.


# --- Main execution ---
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()