# test_ui.py (一個只顯示 UI 的版本)

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QLabel, QHBoxLayout
)
from pyvistaqt.qt_plugin import QtInteractor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI 測試視窗")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        # 左側控制面板
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setFixedWidth(300)

        self.btn_load = QPushButton("1. 載入 STEP 檔案 (已停用)")
        self.btn_analyze = QPushButton("2. 分析顆粒密度 (已停用)")
        self.status_label = QLabel("這是一個 UI 測試。\n如果能看到這個視窗，代表 PyQt6 和 PyVista 安裝正確！")
        self.status_label.setWordWrap(True)
        
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addWidget(self.btn_analyze)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addStretch()

        # 右側 3D 檢視器
        self.plotter_widget = QWidget()
        self.plotter_layout = QVBoxLayout(self.plotter_widget)
        self.plotter = QtInteractor(self.plotter_widget)
        self.plotter_layout.addWidget(self.plotter)

        self.layout.addWidget(self.control_panel)
        self.layout.addWidget(self.plotter_widget, 1)
        
        # 顯示一個簡單的測試物件
        self.plotter.add_mesh(pv.Sphere(), name="test_sphere")
        self.plotter.reset_camera()

if __name__ == "__main__":
    # 確保在 cad_env 環境中運行
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
