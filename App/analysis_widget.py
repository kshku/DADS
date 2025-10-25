"""
Analysis Widget Module
Handles audio analysis, playback controls, and visualization
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtMultimedia import QAudio
from PyQt5.QtGui import QCursor
from plot_canvas import PlotCanvas

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
        
        # Audio Slider
        self.audio_slider = QSlider(Qt.Horizontal)
        self.audio_slider.setObjectName("audioSlider")
        self.audio_slider.setCursor(QCursor(Qt.PointingHandCursor))
        plot_layout.addWidget(self.audio_slider)
        self.audio_slider.hide()
        
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
        
        self.class_prolongation = QLabel("Prolongation: -")
        self.class_repetition = QLabel("Repetition: -")
        self.class_blockage = QLabel("Blockage: -")
        self.class_interjection = QLabel("Interjection: -")
        
        for lbl in [self.class_prolongation, self.class_repetition, 
                    self.class_blockage, self.class_interjection]:
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
                self.audio_slider.show()
                
                self.play_pause_btn.setText("▶ Play")
                self.audio_controls_widget.show()
                self.graph_controls_widget.show()
                self.analysis_canvas.update_progress_line(0)
            except Exception as e:
                print(f"Error during analysis: {e}")
                self.analysis_canvas.clear_plot(f"Error: {e}")
                self.audio_controls_widget.hide()
        else:
            self.analysis_canvas.clear_plot("No audio data to analyze")
            self.audio_controls_widget.hide()
            self.graph_controls_widget.hide()
            self.audio_slider.hide()
            self.current_samples = None
    
    def unload_analysis(self):
        """Clean up analysis view"""
        if self.audio_handler and self.audio_handler.audio_output:
            self.audio_handler.audio_output.stop()
        self.playback_timer.stop()
        if self.audio_handler:
            self.audio_handler.audio_play_buffer.close()
        
        self.current_playback_position_sec = 0.0
        self.playback_start_position_sec = 0.0
        self.is_seeking = False
        self.analysis_canvas.clear_plot()
        self.audio_controls_widget.hide()
        self.graph_controls_widget.hide()
        self.audio_slider.hide()
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
        """Handle slider movement - update visual position"""
        if not self.audio_handler or not self.is_seeking:
            return
        
        # Update visual position immediately
        new_time_sec = position_ms / 1000.0
        self.current_playback_position_sec = new_time_sec
        self.analysis_canvas.update_progress_line(new_time_sec)
    
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
            self.audio_handler.audio_output.reset()
            self.audio_handler.playback_start_time = 0  # Reset internal timer
            self.audio_handler.audio_output.start(self.audio_handler.audio_play_buffer)
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
            # Calculate progress based on buffer position and bytes processed
            bytes_available = self.audio_handler.audio_play_buffer.size()
            bytes_processed = self.audio_handler.audio_play_buffer.pos()
            progress = bytes_processed / bytes_available if bytes_available > 0 else 0
            
            # Calculate current time position
            total_time = self.analysis_canvas.total_time_sec
            self.current_playback_position_sec = progress * total_time
            
            # Clamp to valid range
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
            
            # Check if we reached the end
            if progress >= 1.0:
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
