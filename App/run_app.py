import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main_window import MainWindow
from PyQt5.QtWidgets import QApplication


# --- Main execution ---
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Note: If you encounter 'ModuleNotFoundError', you may need to adjust the path
    # or ensure you run this file from the directory containing the other files.
    main()
