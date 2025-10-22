import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStackedWidget
)
from PyQt5.QtGui import QFont, QCursor
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize variables
        self.audio_file = None
        self.audio_data = None
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("DADS")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QPushButton {
                background-color: #2d3436;
                color: #ffffff;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                min-width: 120px;
                border: 1px solid #404040;
            }
            QPushButton:hover {
                background-color: #353b48;
                border: 1px solid #505050;
            }
            QMainWindow {
                padding: 0;
                margin: 0;
            }
            QLabel {
                color: #ffffff;
            }
            QLabel#mainArea {
                background-color: #2d3436;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #404040;
                margin-right: 5px;
            }
            QLabel#stutterArea {
                background-color: #2d3436;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #404040;
                margin-left: 5px;
                font-size: 14px;
                color: white;
            }
            QFrame#bottomBar {
                background-color: #2f3640;
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
                padding: 10px;
            }
        """)

        # Create main layout with stacked widget
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create stacked widget for switching between views
        self.stack = QStackedWidget()
        
        # Create and add main view
        self.main_page = QWidget()
        self.init_main_page()
        self.stack.addWidget(self.main_page)
        
        # Create and add analysis view
        self.analysis_page = QWidget()
        self.init_analysis_page()
        self.stack.addWidget(self.analysis_page)

        # Add stack to main layout
        main_layout.addWidget(self.stack)

    def on_analysis(self):
        self.stack.setCurrentIndex(1)  # Switch to analysis view
        
    def on_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3)"
        )
        if file_path:
            print(f"Selected file: {file_path}")
            
    def on_analyze(self):
        if not self.is_analyzing:
            # Start analysis
            self.is_analyzing = True
            self.analyze_btn.setText("Stop")
            self.analyze_btn.setStyleSheet("""
                QPushButton {
                    background-color: #c0392b;
                    color: white;
                    border: none;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 14px;
                    min-width: 120px;
                    border: 1px solid #e74c3c;
                }
                QPushButton:hover {
                    background-color: #e74c3c;
                    border: 1px solid #c0392b;
                }
            """)
            print("Starting analysis...")
        else:
            # Stop analysis
            self.is_analyzing = False
            self.analyze_btn.setText("Start")
            self.analyze_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 14px;
                    min-width: 120px;
                    border: 1px solid #2ecc71;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                    border: 1px solid #27ae60;
                }
            """)
            print("Stopping analysis...")

    def init_main_page(self):
        layout = QVBoxLayout()
        self.main_page.setLayout(layout)
        
        # Create title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #16181c;
                min-height: 40px;
                max-height: 40px;
                border-bottom: 1px solid #404040;
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)

        title = QLabel("Stutter Detection System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)

        # Create split content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_widget.setMinimumHeight(350)

        self.main_area = QLabel("Ready to start...")
        self.main_area.setObjectName("mainArea")
        self.main_area.setAlignment(Qt.AlignCenter)

        self.passage_area = QLabel("Select Passage\nNo passage selected")
        self.passage_area.setObjectName("stutterArea")
        self.passage_area.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        
        content_layout.addWidget(self.main_area, 80)
        content_layout.addWidget(self.passage_area, 20)
        layout.addWidget(content_widget)

        # Bottom bar
        bottom_bar = QWidget()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMaximumHeight(80)
        bottom_layout = QHBoxLayout()
        bottom_bar.setLayout(bottom_layout)

        upload_btn = QPushButton("📁")
        self.analyze_btn = QPushButton("Start")
        self.analysis_btn = QPushButton("🔍 Analyze")

        analyze_style = """
            QPushButton {
                background-color: #1b7a44;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #218c4e;
            }
        """
        
        analysis_style = """
            QPushButton {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
        """
        
        self.analyze_btn.setStyleSheet(analyze_style)
        self.analysis_btn.setStyleSheet(analysis_style)

        bottom_layout.addStretch()
        for btn in [upload_btn, self.analyze_btn, self.analysis_btn]:
            bottom_layout.addWidget(btn)
            btn.setCursor(Qt.PointingHandCursor)
        bottom_layout.addStretch()

        upload_btn.clicked.connect(self.on_upload)
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analysis_btn.clicked.connect(self.on_analysis)

        layout.addWidget(bottom_bar)

    def init_analysis_page(self):
        layout = QVBoxLayout()
        self.analysis_page.setLayout(layout)
        
        # Create title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #16181c;
                min-height: 40px;
                max-height: 40px;
                border-bottom: 1px solid #404040;
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)

        title = QLabel("Analysis View")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        layout.addWidget(title_bar)

        # Create split content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_widget.setMinimumHeight(350)

        analysis_area = QLabel("Analysis Content")
        analysis_area.setObjectName("mainArea")
        analysis_area.setAlignment(Qt.AlignCenter)

        passage_area = QLabel("Stutter Classes Detected\nNone detected")
        passage_area.setObjectName("stutterArea")
        passage_area.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        
        content_layout.addWidget(analysis_area, 80)
        content_layout.addWidget(passage_area, 20)
        layout.addWidget(content_widget)

        # Bottom bar with back button
        bottom_bar = QWidget()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMaximumHeight(80)
        bottom_layout = QHBoxLayout()
        bottom_bar.setLayout(bottom_layout)

        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
        """)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.on_back)

        bottom_layout.addWidget(back_btn)
        bottom_layout.addStretch()

        layout.addWidget(bottom_bar)

    def on_back(self):
        self.stack.setCurrentIndex(0)  # Switch back to main view

    def on_analysis(self):
        self.stack.setCurrentIndex(1)  # Switch to analysis view

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
