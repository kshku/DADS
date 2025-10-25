"""
PDF Viewer Widget Module
Handles PDF display, navigation, and zoom functionality
"""
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea,
    QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

class PDFViewerWidget(QWidget):
    """Widget for viewing and managing PDF files"""
    
    pdf_status_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pdf_files = {}
        self.current_pdf_doc = None
        self.current_pdf_page = 0
        self.pdf_zoom = 1.0
        self.pdf_fit_to_page = True
        self._init_ui()
    
    def _init_ui(self):
        """Initialize PDF viewer UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # PDF Scroll Area
        self.pdf_scroll_area = QScrollArea()
        self.pdf_scroll_area.setObjectName("pdfScrollArea")
        self.pdf_scroll_area.setWidgetResizable(True)
        self.pdf_scroll_area.setAlignment(Qt.AlignCenter)
        self.pdf_scroll_area.setMinimumSize(400, 400)
        
        self.pdf_label = QLabel("Select a PDF or start a new session.")
        self.pdf_label.setObjectName("pdfLabel")
        self.pdf_label.setStyleSheet("background-color: #1e1e1e; border: none; padding: 0;")
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_label.setWordWrap(True)
        
        self.pdf_scroll_area.setWidget(self.pdf_label)
        layout.addWidget(self.pdf_scroll_area, 1)
        
        # Navigation Widget
        self.pdf_nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.pdf_nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        
        # Zoom buttons
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setObjectName("zoomButton")
        self.zoom_reset_btn = QPushButton("Fit Page")
        self.zoom_reset_btn.setObjectName("zoomFitButton")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setObjectName("zoomButton")
        
        # Page navigation
        self.prev_page_btn = QPushButton("< Prev")
        self.prev_page_btn.setObjectName("prevPageButton")
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.next_page_btn = QPushButton("Next >")
        self.next_page_btn.setObjectName("nextPageButton")
        
        nav_layout.addStretch()
        nav_layout.addWidget(self.zoom_out_btn)
        nav_layout.addWidget(self.zoom_reset_btn)
        nav_layout.addWidget(self.zoom_in_btn)
        nav_layout.addSpacing(30)
        nav_layout.addWidget(self.prev_page_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_page_btn)
        nav_layout.addStretch()
        
        layout.addWidget(self.pdf_nav_widget)
        self.pdf_nav_widget.hide()
        
        # Connections
        self.zoom_in_btn.clicked.connect(lambda: self.on_zoom(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self.on_zoom(0.8))
        self.zoom_reset_btn.clicked.connect(self.on_zoom_fit)
        self.prev_page_btn.clicked.connect(self.on_prev_page)
        self.next_page_btn.clicked.connect(self.on_next_page)
    
    def on_pdf_item_clicked(self, item):
        """Load and render selected PDF"""
        file_name = item.text()
        file_path = self.pdf_files.get(file_name)
        if not file_path:
            return
            
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
        
        try:
            self.current_pdf_doc = fitz.open(file_path)
            self.current_pdf_page = 0
            self.pdf_fit_to_page = True
            self.pdf_scroll_area.setWidgetResizable(True)
            self.render_pdf_page()
            self.pdf_nav_widget.show()
            self.pdf_status_changed.emit(f"Viewing: {file_name}")
        except Exception as e:
            self.pdf_label.clear()
            self.pdf_label.setText("Error opening PDF.")
            self.pdf_label.setAlignment(Qt.AlignCenter)
            self.pdf_status_changed.emit(f"Error opening PDF: {e}")
            self.pdf_nav_widget.hide()
            self.current_pdf_doc = None
    
    def on_prev_page(self):
        """Navigate to previous PDF page"""
        if self.current_pdf_doc and self.current_pdf_page > 0:
            self.current_pdf_page -= 1
            self.render_pdf_page()
    
    def on_next_page(self):
        """Navigate to next PDF page"""
        if self.current_pdf_doc and self.current_pdf_page < self.current_pdf_doc.page_count - 1:
            self.current_pdf_page += 1
            self.render_pdf_page()
    
    def on_zoom(self, factor):
        """Apply zoom factor"""
        if not self.current_pdf_doc:
            return
        self.pdf_fit_to_page = False
        self.pdf_zoom *= factor
        self.pdf_zoom = max(0.1, min(self.pdf_zoom, 8.0))
        self.render_pdf_page()
    
    def on_zoom_fit(self):
        """Reset to fit-to-page mode"""
        if not self.current_pdf_doc:
            return
        self.pdf_fit_to_page = True
        self.render_pdf_page()
    
    def render_pdf_page(self):
        """Render current PDF page"""
        if not self.current_pdf_doc:
            return
        
        try:
            page = self.current_pdf_doc.load_page(self.current_pdf_page)
            zoom = 1.0
            
            if self.pdf_fit_to_page:
                self.pdf_scroll_area.setWidgetResizable(True)
                view_width = self.pdf_scroll_area.viewport().width() - 10
                view_height = self.pdf_scroll_area.viewport().height() - 10
                
                if view_width <= 0 or view_height <= 0:
                    return
                
                page_rect = page.rect
                zoom_x = view_width / page_rect.width
                zoom_y = view_height / page_rect.height
                zoom = min(zoom_x, zoom_y)
                self.pdf_zoom = zoom
            else:
                self.pdf_scroll_area.setWidgetResizable(False)
                zoom = self.pdf_zoom
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            q_pixmap = QPixmap.fromImage(q_image)
            
            self.pdf_label.setPixmap(q_pixmap)
            self.pdf_label.adjustSize()
            self.pdf_label.setAlignment(Qt.AlignCenter)
            
            self.page_label.setText(f"Page {self.current_pdf_page + 1} / {self.current_pdf_doc.page_count}")
            self.prev_page_btn.setEnabled(self.current_pdf_page > 0)
            self.next_page_btn.setEnabled(self.current_pdf_page < self.current_pdf_doc.page_count - 1)
        except Exception as e:
            self.pdf_label.clear()
            self.pdf_label.setText("Error rendering PDF page.")
            self.pdf_label.setAlignment(Qt.AlignCenter)
            self.pdf_status_changed.emit(f"Error rendering page: {e}")
    
    def clear_pdf_view(self):
        """Reset PDF viewer"""
        if self.current_pdf_doc:
            self.current_pdf_doc.close()
            self.current_pdf_doc = None
        
        self.pdf_nav_widget.hide()
        self.pdf_label.clear()
        self.pdf_label.setText("Select a PDF or start a new session.")
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_fit_to_page = True
        self.pdf_scroll_area.setWidgetResizable(True)
        self.pdf_zoom = 1.0
    
    def add_pdf_file(self, name, path):
        """Add PDF file to the list"""
        self.pdf_files[name] = path
