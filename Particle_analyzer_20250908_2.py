# 檔名: particle_analyzer_app.py
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QMessageBox
)
from pyvistaqt import QtInteractor
from ocp_cat_viewer import build_assembly, get_vertices_from_assembly
from scipy.stats import gaussian_kde

# --- 主應用程式視窗 ---
class ParticleAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("互動式顆粒分析工具 (v2025.09.08)")
        self.setGeometry(100, 100, 1200, 800)

        # --- 變數初始化 ---
        self.assembly = None  # 用於儲存載入的 3D 模型
        self.particle_vertices = None # 用於儲存模型的頂點 (顆粒)
        self.density_volume = None # 用於儲存密度分析結果

        # --- 設定主視窗介面 ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左側控制面板 ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)

        self.btn_load = QPushButton("1. 載入 STEP 檔案")
        self.btn_load.clicked.connect(self.load_step_file)

        self.btn_analyze = QPushButton("2. 分析顆粒密度")
        self.btn_analyze.clicked.connect(self.visualize_density)
        self.btn_analyze.setEnabled(False) # 初始為不可用

        self.btn_reset = QPushButton("重設視圖")
        self.btn_reset.clicked.connect(self.reset_view)
        self.btn_reset.setEnabled(False) # 初始為不可用

        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_analyze)
        control_layout.addWidget(self.btn_reset)
        control_layout.addStretch() # 將按鈕推到頂部

        # --- 右側 3D 顯示區域 ---
        self.plotter = QtInteractor(self)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.plotter, 1) # 讓 plotter 佔用更多空間

        # 初始化 3D 視圖
        self.plotter.add_axes()
        self.plotter.add_grid()

    def load_step_file(self):
        """彈出檔案對話框，讓使用者選擇 STEP 檔案並載入。"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "選擇一個 STEP 或 STP 檔案",
            "", # 預設目錄
            "STEP Files (*.step *.stp);;All Files (*)"
        )

        if not filepath:
            # 使用者取消選擇
            return

        print(f"正在載入檔案: {filepath}")
        try:
            # 清除舊模型
            self.reset_view()

            # 使用 ocp-cat-viewer 讀取檔案
            self.assembly = build_assembly(filepath, name=os.path.basename(filepath))
            
            # 從模型中提取所有頂點
            self.particle_vertices = get_vertices_from_assembly(self.assembly)

            if self.particle_vertices is None or len(self.particle_vertices) == 0:
                QMessageBox.warning(self, "載入錯誤", "無法從模型中提取任何頂點(顆粒)！")
                return

            # 在 3D 視窗中顯示模型
            self.plotter.add_assembly(self.assembly, name="original_model")
            self.plotter.reset_camera()
            
            QMessageBox.information(self, "載入成功", 
                f"成功載入模型 '{os.path.basename(filepath)}'。\n"
                f"共偵測到 {len(self.particle_vertices)} 個顆粒 (頂點)。\n"
                "現在可以點擊按鈕進行密度分析。")

            # 啟用後續按鈕
            self.btn_analyze.setEnabled(True)
            self.btn_reset.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "載入失敗", f"讀取 STEP 檔案時發生錯誤:\n{e}")
            self.reset_view()

    def visualize_density(self):
        """計算並視覺化顆粒的空間分佈密度。"""
        if self.particle_vertices is None:
            QMessageBox.warning(self, "操作錯誤", "請先載入一個包含顆粒的 STEP 檔案。")
            return

        try:
            print("開始計算顆粒密度...")
            # 清除之前的密度雲
            if self.density_volume:
                self.plotter.remove_actor(self.density_volume)

            coords = self.particle_vertices.T # Scipy KDE 需要 (維度, 點數) 的格式

            # 1. 使用 Scipy 進行核密度估計 (KDE)
            kde = gaussian_kde(coords)

            # 2. 建立一個包圍所有點的 3D 網格
            x_min, y_min, z_min = self.particle_vertices.min(axis=0)
            x_max, y_max, z_max = self.particle_vertices.max(axis=0)
            
            # 定義網格解析度
            dim = 50 
            xi, yi, zi = np.mgrid[x_min:x_max:complex(dim), y_min:y_max:complex(dim), z_min:z_max:complex(dim)]

            # 3. 在網格的每個點上計算密度
            grid_coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
            density = kde(grid_coords).reshape(xi.shape)

            # 4. 使用 PyVista 建立結構化網格物件
            grid = pv.StructuredGrid(xi, yi, zi)
            grid["density"] = density.ravel(order="F") # 將密度值附加到網格

            # 5. 在 plotter 中以體積形式渲染密度
            # opacity 設為一個漸變，讓低密度區域更透明
            self.density_volume = self.plotter.add_volume(
                grid,
                cmap="magma",
                opacity="linear",
                shade=True,
                name="density_cloud"
            )
            
            self.plotter.add_text("顆粒空間分佈密度\n(顏色越亮密度越高)", font_size=12)
            
            # 隱藏原始模型以突顯密度雲
            self.plotter.remove_actor("original_model", render=False)
            self.plotter.reset_camera()
            
            QMessageBox.information(self, "分析完成", "已在視窗中顯示空間分佈密度雲。")

        except Exception as e:
            QMessageBox.critical(self, "分析失敗", f"計算空間分佈時發生錯誤:\n{e}")

    def reset_view(self):
        """清除所有分析結果，並重新顯示原始模型。"""
        self.plotter.clear() # 清除所有 actor
        self.density_volume = None
        
        # 重新加入座標軸和網格
        self.plotter.add_axes()
        self.plotter.add_grid()

        # 如果原始模型存在，則重新顯示它
        if self.assembly:
            self.plotter.add_assembly(self.assembly, name="original_model")
        
        self.plotter.reset_camera()
        print("視圖已重設。")


# --- 程式主入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParticleAnalyzerApp()
    window.show()
    sys.exit(app.exec())
