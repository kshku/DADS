"""
Analysis Widget Module
Handles audio analysis, playback controls, and visualization
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSlider, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSignal as Signal
from PyQt5.QtMultimedia import QAudio
from PyQt5.QtGui import QCursor
from plot_canvas import PlotCanvas
from connector import StutterDetector
import tempfile
import os
from datetime import datetime


class StutterDetectionThread(QThread):
    """Background thread for running stutter detection"""
    detection_complete = Signal(dict)
    detection_error = Signal(str)
    progress_update = Signal(int, int, dict)  # chunk_idx, total_chunks, aggregated_results
    
    def __init__(self, audio_data, sample_rate, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.temp_file = None
        self.aggregated_results = {}
        self.chunk_detections = {}  # Store detections per chunk
    
    def progress_callback(self, chunk_idx, total_chunks, chunk_results):
        """Callback for progress updates from detector"""
        # Store chunk results
        self.chunk_detections[chunk_idx] = {'detections': chunk_results}
        
        # Count how many detections so far
        detected_counts = {}
        for stutter_type in ['prolongation', 'soundrep', 'wordrep', 'block', 'interjection']:
            count = 0
            for chunk_data in self.chunk_detections.values():
                if ('detections' in chunk_data and 
                    stutter_type in chunk_data['detections'] and 
                    chunk_data['detections'][stutter_type]['detected']):
                    count += 1
            detected_counts[stutter_type] = count
        
        # Calculate aggregated statistics 
        aggregated = {}
        processed_chunks = len(self.chunk_detections)
        
        for stutter_type in ['prolongation', 'soundrep', 'wordrep', 'block', 'interjection']:
            total_prob = 0.0
            max_prob = 0.0
            detected_count = detected_counts[stutter_type]
            
            for chunk_data in self.chunk_detections.values():
                if 'detections' in chunk_data and stutter_type in chunk_data['detections']:
                    prob = chunk_data['detections'][stutter_type]['probability']
                    total_prob += prob
                    max_prob = max(max_prob, prob)
            
            avg_confidence = (total_prob / processed_chunks * 100) if processed_chunks > 0 else 0.0
            detected = max_prob > 0.4
            
            aggregated[stutter_type] = {
                'confidence': avg_confidence,
                'detected': detected,
                'max_probability': max_prob,
                'detected_chunks': detected_count
            }
        
        # Emit progress update
        self.progress_update.emit(chunk_idx, total_chunks, aggregated)
    
    def run(self):
        """Run stutter detection in background"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                self.temp_file = tmp.name
                
                # Write WAV file
                import wave
                with wave.open(self.temp_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(self.audio_data.tobytes())
            
            # Initialize detector and process
            detector = StutterDetector(detection_threshold=0.4)
            
            # Process with progress callback
            results = detector.process_audio_file(self.temp_file, callback=self.progress_callback)
            
            # Clean up temp file
            if self.temp_file and os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            
            self.detection_complete.emit(results)
            
        except Exception as e:
            if self.temp_file and os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            self.detection_error.emit(str(e))


class AnalysisWidget(QWidget):
    """Widget for analyzing audio and controlling playback"""
    
    back_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_handler = None
        self.current_graph_type = 'spectrogram'
        self.current_samples = None
        self.playback_timer = QTimer(self)
        self.current_playback_position_sec = 0.0
        self.playback_start_position_sec = 0.0
        self.was_playing_before_drag = False
        self.is_seeking = False
        self.detection_thread = None
        self.detection_results = None
        
        # Stutter detection counters - persistent across audio files
        self.stutter_counts = {
            'prolongation': 0,
            'soundrep': 0,
            'wordrep': 0,
            'block': 0,
            'interjection': 0
        }
        
        self._init_ui()
        self.playback_timer.timeout.connect(self.update_playback_progress)
        self.playback_timer.setInterval(100)  # Update every 100ms for smoother visuals
    
    def set_audio_handler(self, handler):
        """Set the audio handler reference"""
        self.audio_handler = handler
    
    def _init_ui(self):
        """Initialize analysis widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title Bar
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("Analysis View")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)
        
        # Content Area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)
        
        # Plot Container
        analysis_plot_container = QWidget()
        analysis_plot_container.setStyleSheet(
            "background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333333; padding: 5px;"
        )
        plot_layout = QVBoxLayout(analysis_plot_container)
        plot_layout.setContentsMargins(5, 5, 5, 5)
        plot_layout.setSpacing(10)
        
        # Graph Controls
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
        
        plot_layout.addWidget(self.graph_controls_widget)
        self.graph_controls_widget.hide()
        
        # Matplotlib canvas
        self.analysis_canvas = PlotCanvas(analysis_plot_container)
        plot_layout.addWidget(self.analysis_canvas, 1)
        
        # Audio Slider with time label
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(10)
        
        self.audio_slider = QSlider(Qt.Horizontal)
        self.audio_slider.setObjectName("audioSlider")
        self.audio_slider.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Time label to show current position
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("color: #ffffff; font-size: 12px; min-width: 100px;")
        self.time_label.setAlignment(Qt.AlignCenter)
        
        slider_layout.addWidget(self.audio_slider)
        slider_layout.addWidget(self.time_label)
        
        plot_layout.addWidget(slider_container)
        slider_container.hide()
        self.slider_container = slider_container
        
        # Audio Controls
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
        
        plot_layout.addWidget(self.audio_controls_widget)
        self.audio_controls_widget.hide()
        
        # Stutter Panel
        self.stutter_panel = QWidget()
        self.stutter_panel.setObjectName("stutterPanel")
        self.stutter_panel.setStyleSheet("""
            QWidget#stutterPanel {
                background-color: #1e1e1e;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #333333;
            }
        """)
        stutter_layout = QVBoxLayout(self.stutter_panel)
        
        title_label = QLabel("Stutter Classes Detected")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        stutter_layout.addWidget(title_label)
        
        # Status label for detection progress
        self.detection_status_label = QLabel("Analyzing...")
        self.detection_status_label.setStyleSheet(
            "font-size: 14px; padding: 10px; color: #ffa500; "
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        self.detection_status_label.setAlignment(Qt.AlignCenter)
        stutter_layout.addWidget(self.detection_status_label)
        self.detection_status_label.hide()
        
        self.class_prolongation = QLabel("Prolongation: - [0]")
        self.class_soundrep = QLabel("Sound Repetition: - [0]")
        self.class_wordrep = QLabel("Word Repetition: - [0]")
        self.class_block = QLabel("Block: - [0]")
        self.class_interjection = QLabel("Interjection: - [0]")
        
        for lbl in [self.class_prolongation, self.class_soundrep, self.class_wordrep,
                    self.class_block, self.class_interjection]:
            lbl.setStyleSheet("font-size: 16px; padding: 5px 0; color: #e0e0e0;")
            stutter_layout.addWidget(lbl)
        
        stutter_layout.addStretch(1)
        
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.setObjectName("exportButton")
        self.export_report_btn.setCursor(QCursor(Qt.PointingHandCursor))
        stutter_layout.addWidget(self.export_report_btn)
        
        content_layout.addWidget(analysis_plot_container, 75)
        content_layout.addWidget(self.stutter_panel, 25)
        layout.addWidget(content_widget, 1)
        
        # Bottom Bar
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
        
        # Connections
        self.back_btn.clicked.connect(self.back_requested.emit)
        self.play_pause_btn.clicked.connect(self.on_play_pause)
        self.spec_btn.clicked.connect(lambda: self.on_graph_type_changed('spectrogram'))
        self.wave_btn.clicked.connect(lambda: self.on_graph_type_changed('waveform'))
        self.export_report_btn.clicked.connect(self.export_report)
        self.audio_slider.sliderPressed.connect(self.on_slider_pressed)
        self.audio_slider.sliderMoved.connect(self.on_slider_moved)
        self.audio_slider.sliderReleased.connect(self.on_slider_released)
        self.back_btn_5s.clicked.connect(lambda: self.seek_to_position(max(0, self.audio_slider.value() - 5000)))
        self.fwd_btn_5s.clicked.connect(lambda: self.seek_to_position(min(self.audio_slider.maximum(), self.audio_slider.value() + 5000)))
        
        self.analysis_canvas.clear_plot()
    
    def load_analysis(self):
        """Load and display audio analysis"""
        if not self.audio_handler:
            return
        
        audio_data = self.audio_handler.get_audio_data()
        sample_rate = self.audio_handler.get_sample_rate()
        
        if len(audio_data) > 0 and sample_rate > 0:
            try:
                self.current_samples = np.frombuffer(audio_data, dtype=np.int16)
                self.current_graph_type = 'spectrogram'
                self.draw_current_graph()
                
                if not self.audio_handler.setup_playback():
                    raise RuntimeError("Could not initialize audio output.")
                
                # Reset playback state
                self.current_playback_position_sec = 0.0
                self.playback_start_position_sec = 0.0
                self.is_seeking = False
                
                # Setup slider
                total_time_ms = int(self.analysis_canvas.total_time_sec * 1000)
                self.audio_slider.setRange(0, total_time_ms)
                self.audio_slider.setValue(0)
                self.slider_container.show()
                
                # Update time label
                self.update_time_label(0, self.analysis_canvas.total_time_sec)
                
                self.play_pause_btn.setText("▶ Play")
                self.audio_controls_widget.show()
                self.graph_controls_widget.show()
                self.analysis_canvas.update_progress_line(0)
                
                # Start stutter detection in background
                self.start_stutter_detection()
                
            except Exception as e:
                print(f"Error during analysis: {e}")
                self.analysis_canvas.clear_plot(f"Error: {e}")
                self.audio_controls_widget.hide()
        else:
            self.analysis_canvas.clear_plot("No audio data to analyze")
            self.audio_controls_widget.hide()
            self.graph_controls_widget.hide()
            self.slider_container.hide()
            self.current_samples = None
    
    def start_stutter_detection(self):
        """Start stutter detection in background thread"""
        if self.current_samples is None:
            return
        
        # Reset counters for new audio file analysis
        self.stutter_counts = {
            'prolongation': 0,
            'soundrep': 0,
            'wordrep': 0,
            'block': 0,
            'interjection': 0
        }
        
        # Show detection status
        self.detection_status_label.setText("Analyzing stutters...")
        self.detection_status_label.show()
        
        # Reset labels to show analyzing state
        self.class_prolongation.setText("Prolongation: Analyzing...")
        self.class_soundrep.setText("Sound Repetition: Analyzing...")
        self.class_wordrep.setText("Word Repetition: Analyzing...")
        self.class_block.setText("Block: Analyzing...")
        self.class_interjection.setText("Interjection: Analyzing...")
        
        # Create and start detection thread
        sample_rate = self.audio_handler.get_sample_rate()
        self.detection_thread = StutterDetectionThread(self.current_samples, sample_rate)
        self.detection_thread.detection_complete.connect(self.on_detection_complete)
        self.detection_thread.detection_error.connect(self.on_detection_error)
        self.detection_thread.progress_update.connect(self.on_detection_progress)
        self.detection_thread.start()
    
    def on_detection_progress(self, chunk_idx, total_chunks, aggregated_results):
        """Handle progress updates during detection"""
        # Update status label with progress
        progress_pct = ((chunk_idx + 1) / total_chunks) * 100
        self.detection_status_label.setText(f"Analyzing... {progress_pct:.0f}%")
        
        # Update labels with current aggregated results
        for stutter_type, result in aggregated_results.items():
            confidence = result['confidence']
            detected = result['detected']
            detected_chunks = result.get('detected_chunks', 0)
            
            # Update the count in the widget
            self.stutter_counts[stutter_type] = detected_chunks
            
            # Format display text: show confidence percentage
            status_text = f"{confidence:.1f}%"
            if detected:
                status_text += " ✓"
            status_text += f" [{detected_chunks}]"
            
            # Set color
            color = "#00ff00" if detected else "#888888"
            
            # Update appropriate label
            if stutter_type == 'prolongation':
                self.class_prolongation.setText(f"Prolongation: {status_text}")
                self.class_prolongation.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'soundrep':
                self.class_soundrep.setText(f"Sound Repetition: {status_text}")
                self.class_soundrep.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'wordrep':
                self.class_wordrep.setText(f"Word Repetition: {status_text}")
                self.class_wordrep.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'block':
                self.class_block.setText(f"Block: {status_text}")
                self.class_block.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'interjection':
                self.class_interjection.setText(f"Interjection: {status_text}")
                self.class_interjection.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
    
    def on_detection_complete(self, results):
        """Handle completed stutter detection"""
        self.detection_results = results
        self.detection_status_label.setText("Analysis Complete!")
        
        # Calculate overall statistics across all chunks
        total_chunks = len(results)
        aggregated_stats = {}
        
        for stutter_type in ['prolongation', 'soundrep', 'wordrep', 'block', 'interjection']:
            total_prob = 0.0
            max_prob = 0.0
            detected_chunks_count = 0
            
            for chunk_idx, chunk_data in results.items():
                if 'detections' in chunk_data and stutter_type in chunk_data['detections']:
                    detection = chunk_data['detections'][stutter_type]
                    prob = detection['probability']
                    is_detected = detection['detected']
                    
                    total_prob += prob
                    max_prob = max(max_prob, prob)
                    
                    # Count each chunk where this stutter was detected
                    if is_detected and prob > 0.4:  # Only count if it exceeds detection threshold
                        detected_chunks_count += 1
            
            # Calculate average confidence across all chunks
            avg_confidence = (total_prob / total_chunks * 100) if total_chunks > 0 else 0.0
            
            # Detected if max probability exceeds threshold (40%)
            detected = max_prob > 0.4
            
            # Set counter to the number of chunks where detected
            if stutter_type in self.stutter_counts:
                self.stutter_counts[stutter_type] = detected_chunks_count
            
            aggregated_stats[stutter_type] = {
                'confidence': avg_confidence,
                'detected': detected,
                'max_probability': max_prob,
                'detected_chunks': detected_chunks_count
            }
        
        # Update labels with final results
        for stutter_type, stats in aggregated_stats.items():
            confidence = stats['confidence']
            detected = stats['detected']
            detected_chunks = stats['detected_chunks']
            
            # Get the updated count (after setting)
            count = self.stutter_counts.get(stutter_type, 0)
            
            # Format: "Type: XX.X% ✓ [count]"
            status_text = f"{confidence:.1f}%"
            if detected:
                status_text += " ✓"
            status_text += f" [{count}]"
            
            color = "#00ff00" if detected else "#888888"
            
            if stutter_type == 'prolongation':
                self.class_prolongation.setText(f"Prolongation: {status_text}")
                self.class_prolongation.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'soundrep':
                self.class_soundrep.setText(f"Sound Repetition: {status_text}")
                self.class_soundrep.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'wordrep':
                self.class_wordrep.setText(f"Word Repetition: {status_text}")
                self.class_wordrep.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'block':
                self.class_block.setText(f"Block: {status_text}")
                self.class_block.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
            elif stutter_type == 'interjection':
                self.class_interjection.setText(f"Interjection: {status_text}")
                self.class_interjection.setStyleSheet(f"font-size: 16px; padding: 5px 0; color: {color};")
        
        # Hide status label after a delay
        QTimer.singleShot(2000, self.detection_status_label.hide)
    
    def on_detection_error(self, error_msg):
        """Handle detection error"""
        self.detection_status_label.setText(f"Detection Error")
        self.detection_status_label.setStyleSheet(
            "font-size: 14px; padding: 10px; color: #ff5555; "
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        
        # Reset labels to show error with current counts
        error_text = "Error"
        for stutter_type in ['prolongation', 'soundrep', 'wordrep', 'block', 'interjection']:
            count = self.stutter_counts.get(stutter_type, 0)
            
            if stutter_type == 'prolongation':
                self.class_prolongation.setText(f"Prolongation: {error_text} [{count}]")
            elif stutter_type == 'soundrep':
                self.class_soundrep.setText(f"Sound Repetition: {error_text} [{count}]")
            elif stutter_type == 'wordrep':
                self.class_wordrep.setText(f"Word Repetition: {error_text} [{count}]")
            elif stutter_type == 'block':
                self.class_block.setText(f"Block: {error_text} [{count}]")
            elif stutter_type == 'interjection':
                self.class_interjection.setText(f"Interjection: {error_text} [{count}]")
        
        print(f"Stutter detection error: {error_msg}")
    
    def unload_analysis(self):
        """Clean up analysis view"""
        # Stop detection thread if running
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait()
        
        if self.audio_handler and self.audio_handler.audio_output:
            self.audio_handler.audio_output.stop()
        self.playback_timer.stop()
        if self.audio_handler:
            self.audio_handler.audio_play_buffer.close()
        
        self.current_playback_position_sec = 0.0
        self.playback_start_position_sec = 0.0
        self.is_seeking = False
        self.detection_results = None
        
        # Reset stutter labels with counts preserved
        self.class_prolongation.setText(f"Prolongation: - [{self.stutter_counts['prolongation']}]")
        self.class_soundrep.setText(f"Sound Repetition: - [{self.stutter_counts['soundrep']}]")
        self.class_wordrep.setText(f"Word Repetition: - [{self.stutter_counts['wordrep']}]")
        self.class_block.setText(f"Block: - [{self.stutter_counts['block']}]")
        self.class_interjection.setText(f"Interjection: - [{self.stutter_counts['interjection']}]")
        self.detection_status_label.hide()
        
        self.analysis_canvas.clear_plot()
        self.audio_controls_widget.hide()
        self.graph_controls_widget.hide()
        self.slider_container.hide()
        self.current_samples = None
    
    def draw_current_graph(self):
        """Draw the currently selected graph type"""
        if self.current_samples is None:
            return
        
        sample_rate = self.audio_handler.get_sample_rate()
        if self.current_graph_type == 'spectrogram':
            self.analysis_canvas.plot_spectrogram(self.current_samples, sample_rate)
            self.spec_btn.setProperty('selected', True)
            self.wave_btn.setProperty('selected', False)
        elif self.current_graph_type == 'waveform':
            self.analysis_canvas.plot_waveform(self.current_samples, sample_rate)
            self.spec_btn.setProperty('selected', False)
            self.wave_btn.setProperty('selected', True)
        
        self.spec_btn.style().unpolish(self.spec_btn)
        self.spec_btn.style().polish(self.spec_btn)
        self.wave_btn.style().unpolish(self.wave_btn)
        self.wave_btn.style().polish(self.wave_btn)
    
    def on_graph_type_changed(self, graph_type):
        """Switch graph type"""
        if graph_type == self.current_graph_type or self.current_samples is None:
            return
        
        self.current_graph_type = graph_type
        self.draw_current_graph()
        
                # Update progress line to current position
        self.analysis_canvas.update_progress_line(self.current_playback_position_sec)
    
    def on_slider_pressed(self):
        """Handle slider press - pause playback during drag"""
        if not self.audio_handler or not self.audio_handler.audio_output:
            return
        
        # Remember if we were playing
        self.was_playing_before_drag = (self.audio_handler.audio_output.state() == QAudio.ActiveState)
        
        # Stop playback and timer
        if self.was_playing_before_drag:
            self.audio_handler.audio_output.stop()
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        
        self.is_seeking = True
    
    def on_slider_moved(self, position_ms):
        """Handle slider movement - update visual position and time label"""
        if not self.audio_handler or not self.is_seeking:
            return
        
        # Update visual position immediately
        new_time_sec = position_ms / 1000.0
        self.current_playback_position_sec = new_time_sec
        
        # Update the graph window if needed
        self.analysis_canvas.update_progress_line(new_time_sec)
        
        # Update time label
        self.update_time_label(new_time_sec, self.analysis_canvas.total_time_sec)
    
    def on_slider_released(self):
        """Handle slider release - seek to new position"""
        if not self.audio_handler or not self.is_seeking:
            return
        
        self.is_seeking = False
        new_time_sec = self.audio_slider.value() / 1000.0
        
        # Perform the actual seek
        self.seek_to_time(new_time_sec, force_play_state=self.was_playing_before_drag)
    
    
    def on_play_pause(self):
        """Toggle between play and pause states"""
        if not self.audio_handler or not self.audio_handler.audio_output:
            return
        
        state = self.audio_handler.audio_output.state()
        
        if state == QAudio.ActiveState:
            # Currently playing - pause it
            self.audio_handler.audio_output.stop()
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        else:
            # Currently paused or stopped - start/resume playback
            # Seek to current position and start playing
            self.seek_to_time(self.current_playback_position_sec, force_play_state=True)
    
    def seek_to_position(self, position_ms):
        """Handle scrubber seek requests."""
        if not self.audio_handler or not self.audio_handler.audio_output:
            return
        resume_playing = (self.audio_handler.audio_output.state() == QAudio.ActiveState)
        new_time_sec = position_ms / 1000.0
        self.seek_to_time(new_time_sec, force_play_state=resume_playing)
    
    def seek_to_time(self, new_time_sec, force_play_state=None):
        """Seek to specific time in audio"""
        if not self.audio_handler or not self.audio_handler.audio_output:
            return
        
        # Stop current playback
        was_playing = (self.audio_handler.audio_output.state() == QAudio.ActiveState)
        self.audio_handler.audio_output.stop()
        self.playback_timer.stop()
        
        # Clamp time to valid range
        new_time_sec = max(0, min(new_time_sec, self.analysis_canvas.total_time_sec))
        
        # Calculate byte position
        new_pos_bytes = int(new_time_sec * self.audio_handler.bytes_per_second)
        # Align to sample boundary (2 bytes for 16-bit audio)
        new_pos_bytes = (new_pos_bytes // 2) * 2
        
        # Seek buffer
        if not self.audio_handler.audio_play_buffer.seek(new_pos_bytes):
            print(f"Warning: Failed to seek to {new_pos_bytes} bytes")
        
        # Update our tracking position
        self.playback_start_position_sec = new_time_sec
        self.current_playback_position_sec = new_time_sec
        
        # Update visuals
        self.analysis_canvas.update_progress_line(new_time_sec)
        self.audio_slider.blockSignals(True)
        self.audio_slider.setValue(int(new_time_sec * 1000))
        self.audio_slider.blockSignals(False)
        
        # Determine whether to resume playback
        should_play = was_playing if force_play_state is None else force_play_state
        
        if should_play:
            # Reset audio output to clear any buffered data
            self.audio_handler.audio_output.reset()
            # Start playback from the new position
            self.audio_handler.audio_output.start(self.audio_handler.audio_play_buffer)
            # Start the timer to update progress
            self.playback_timer.start()
            self.play_pause_btn.setText("❚❚ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
    
    def update_playback_progress(self):
        """Update playback progress - called by timer"""
        if not self.audio_handler or not self.audio_handler.audio_output:
            return
        
        state = self.audio_handler.audio_output.state()
        
        if state == QAudio.ActiveState and not self.is_seeking:
            # Use actual elapsed time from audio output
            elapsed_microseconds = self.audio_handler.audio_output.processedUSecs()
            elapsed_seconds = elapsed_microseconds / 1_000_000.0
            
            # Calculate current position based on where we started + elapsed time
            self.current_playback_position_sec = self.playback_start_position_sec + elapsed_seconds
            
            # Clamp to valid range
            total_time = self.analysis_canvas.total_time_sec
            self.current_playback_position_sec = max(0, min(
                self.current_playback_position_sec, 
                total_time
            ))
            
            # Update visuals
            self.analysis_canvas.update_progress_line(self.current_playback_position_sec)
            
            # Update slider position
            position_ms = int(self.current_playback_position_sec * 1000)
            self.audio_slider.blockSignals(True)
            self.audio_slider.setValue(position_ms)
            self.audio_slider.blockSignals(False)
            
            # Update time label
            self.update_time_label(self.current_playback_position_sec, total_time)
            
            # Check if we reached the end
            if self.current_playback_position_sec >= total_time - 0.1:  # Small buffer for end detection
                self.audio_handler.audio_output.stop()
                self.playback_timer.stop()
                self.play_pause_btn.setText("▶ Play")
        elif state == QAudio.IdleState:
            # Playback finished
            self.playback_timer.stop()
            self.current_playback_position_sec = self.analysis_canvas.total_time_sec
            self.playback_start_position_sec = self.analysis_canvas.total_time_sec
            
            self.analysis_canvas.update_progress_line(self.analysis_canvas.total_time_sec)
            self.audio_slider.blockSignals(True)
            self.audio_slider.setValue(self.audio_slider.maximum())
            self.audio_slider.blockSignals(False)
            
            self.play_pause_btn.setText("▶ Play")
    
    def update_time_label(self, current_sec, total_sec):
        """Update the time label with formatted time"""
        current_min = int(current_sec // 60)
        current_s = int(current_sec % 60)
        total_min = int(total_sec // 60)
        total_s = int(total_sec % 60)
        
        self.time_label.setText(f"{current_min}:{current_s:02d} / {total_min}:{total_s:02d}")
    
    def export_report(self):
        """Export detection results to a text file"""
        if not self.detection_results:
            QMessageBox.warning(
                self,
                "No Results",
                "No detection results available. Please analyze an audio file first."
            )
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get audio file name (if available from audio_handler)
        audio_filename = "unknown_audio"
        if hasattr(self, 'audio_handler') and self.audio_handler:
            # Try to get the file path if available
            audio_filename = getattr(self.audio_handler, 'current_file', 'unknown_audio')
            if audio_filename and audio_filename != 'unknown_audio':
                audio_filename = os.path.basename(audio_filename)
        
        # Default filename suggestion
        report_filename = f"stutter_report_{timestamp}.txt"
        
        # Get default save location (reports directory outside App folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from App/
        default_dir = os.path.join(project_root, "reports")
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, report_filename)
        
        # Open file dialog to let user choose save location
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report As",
            default_path,
            "Text Files (*.txt);;All Files (*)"
        )
        
        # If user cancelled, return
        if not save_path:
            return
        
        report_path = save_path
        
        # Generate report content
        try:
            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("STUTTER DETECTION REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Date and time
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Audio File: {audio_filename}\n")
                
                # Audio file path (if available)
                if hasattr(self, 'audio_handler') and self.audio_handler:
                    full_path = getattr(self.audio_handler, 'current_file', None)
                    if full_path:
                        f.write(f"File Path: {full_path}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("DETECTION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                
                # Calculate overall statistics across all chunks
                total_chunks = len(self.detection_results)
                
                # Aggregate statistics - calculate average confidence
                aggregated_stats = {}
                for stutter_type in ['prolongation', 'soundrep', 'wordrep', 'block', 'interjection']:
                    total_prob = 0.0
                    max_prob = 0.0
                    
                    for chunk_data in self.detection_results.values():
                        if 'detections' in chunk_data and stutter_type in chunk_data['detections']:
                            detection = chunk_data['detections'][stutter_type]
                            prob = detection['probability']
                            total_prob += prob
                            max_prob = max(max_prob, prob)
                    
                    # Average confidence across all chunks
                    avg_confidence = (total_prob / total_chunks * 100) if total_chunks > 0 else 0.0
                    detected = max_prob > 0.4
                    
                    aggregated_stats[stutter_type] = {
                        'confidence': avg_confidence,
                        'detected': detected,
                        'max_probability': max_prob
                    }
                
                f.write("Stutter Classes Detected:\n")
                f.write("-" * 60 + "\n\n")
                
                # Sort by confidence (highest first)
                sorted_stats = sorted(
                    aggregated_stats.items(),
                    key=lambda x: x[1]['confidence'],
                    reverse=True
                )
                
                for stutter_type, stats in sorted_stats:
                    display_name = stutter_type.replace('_', ' ').title()
                    count = self.stutter_counts.get(stutter_type, 0)
                    
                    status = "✓ DETECTED" if stats['detected'] else "○ Not Detected"
                    f.write(f"{display_name:25s}: {stats['confidence']:5.1f}%  {status}  [Count: {count}]\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                # Count detected classes
                detected_count = sum(1 for s in aggregated_stats.values() if s['detected'])
                total_count = len(aggregated_stats)
                
                f.write(f"Total Classes Analyzed: {total_count}\n")
                f.write(f"Classes Detected: {detected_count}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 60 + "\n")
            
            # Show success message
            QMessageBox.information(
                self,
                "Report Exported",
                f"Report successfully exported to:\n{report_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export report:\n{str(e)}"
            )
