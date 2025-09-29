# -*- coding: utf-8 -*-
import sys
import os
import csv
import numpy as np
import trimesh
import pyvista as pv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QSplitter, QGroupBox, QCheckBox, QMessageBox,
    QLabel, QLineEdit
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
from scipy.stats import gaussian_kde
from pyvistaqt import QtInteractor

# 讓 PyVista 與 PySide6 整合
pv.set_plot_theme("document")

class ParticleAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 顆粒形狀與分佈分析工具 (v1.5 - 可調參數版)")
        self.setGeometry(100, 100, 1600, 900)

        # --- 資料儲存 ---
        self.particles = []
        self.particle_data = []
        self.selected_particle_index = -1
        self.scene_mesh = None

        # --- 主介面佈局 ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)

        # --- 左側：全域顆粒視圖 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotter_global = QtInteractor(left_panel)
        left_layout.addWidget(self.plotter_global.interactor)
        left_layout.addWidget(QLabel("全域顆粒分佈視圖 (可點擊選擇顆粒)"))
        
        splitter.addWidget(left_panel)

        # --- 右側：詳細資訊與控制項 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)

        # --- 右側上方：單一顆粒視圖與控制項 ---
        top_right_panel = QWidget()
        top_right_layout = QHBoxLayout(top_right_panel)
        
        detail_view_widget = QWidget()
        detail_view_layout = QVBoxLayout(detail_view_widget)
        self.plotter_detail = QtInteractor(detail_view_widget)
        detail_view_layout.addWidget(self.plotter_detail.interactor)
        detail_view_layout.addWidget(QLabel("單一顆粒詳細視圖"))
        
        # --- 控制項面板 ---
        controls_panel = QWidget()
        controls_panel_layout = QVBoxLayout(controls_panel)
        
        # --- 重要升級：分析參數設定 ---
        params_group = QGroupBox("分析參數")
        params_layout = QVBoxLayout(params_group)
        min_vol_layout = QHBoxLayout()
        min_vol_label = QLabel("最小體積閾值 (nm³):")
        self.min_volume_input = QLineEdit("1e-6") # 預設值設為 0.000001
        self.min_volume_input.setToolTip("體積小於此值的物件將被過濾掉。\n請使用科學記號 (例如 1e-6) 或小數。")
        min_vol_layout.addWidget(min_vol_label)
        min_vol_layout.addWidget(self.min_volume_input)
        params_layout.addLayout(min_vol_layout)
        
        controls_panel_layout.addWidget(params_group)

        # --- 顯示選項 ---
        display_group = QGroupBox("顯示選項")
        display_layout = QVBoxLayout(display_group)
        self.cb_ellipsoid = QCheckBox("擬合橢球")
        self.cb_convex_hull = QCheckBox("凸包 (Convex Hull)")
        self.cb_axes = QCheckBox("主軸")
        
        self.cb_ellipsoid.toggled.connect(self.update_detail_view)
        self.cb_convex_hull.toggled.connect(self.update_detail_view)
        self.cb_axes.toggled.connect(self.update_detail_view)
        
        display_layout.addWidget(self.cb_ellipsoid)
        display_layout.addWidget(self.cb_convex_hull)
        display_layout.addWidget(self.cb_axes)
        
        controls_panel_layout.addWidget(display_group)
        controls_panel_layout.addStretch()

        top_right_layout.addWidget(detail_view_widget, 3)
        top_right_layout.addWidget(controls_panel, 1)

        right_splitter.addWidget(top_right_panel)

        # --- 右側下方：數據表格 ---
        bottom_right_panel = QWidget()
        bottom_right_layout = QVBoxLayout(bottom_right_panel)
        self.table = QTableWidget()
        bottom_right_layout.addWidget(self.table)
        right_splitter.addWidget(bottom_right_panel)
        
        splitter.addWidget(right_panel)
        
        splitter.setSizes([800, 800])
        right_splitter.setSizes([500, 400])

        self.create_actions()
        self.create_menus()
        self.create_toolbar()
        self.reset_app_state()

    # ... (create_actions, create_menus, create_toolbar, reset_app_state 函式與之前相同，為節省篇幅省略) ...
    def create_actions(self):
        self.action_open = QAction(QIcon.fromTheme("document-open"), "開啟 STL 檔案...", self)
        self.action_open.triggered.connect(self.load_stl_file)
        
        self.action_export = QAction(QIcon.fromTheme("document-save"), "匯出數據為 CSV...", self)
        self.action_export.triggered.connect(self.export_to_csv)

        self.action_dist_vis = QAction("可視化空間分佈密度", self)
        self.action_dist_vis.triggered.connect(self.visualize_distribution)

    def create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("檔案")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_export)

        analysis_menu = menu_bar.addMenu("分析")
        analysis_menu.addAction(self.action_dist_vis)

    def create_toolbar(self):
        toolbar = self.addToolBar("主工具列")
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_export)
        toolbar.addAction(self.action_dist_vis)

    def reset_app_state(self):
        self.particles = []
        self.particle_data = []
        self.selected_particle_index = -1
        self.scene_mesh = None
        self.plotter_global.clear()
        self.plotter_detail.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.action_export.setEnabled(False)
        self.action_dist_vis.setEnabled(False)

    def load_stl_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "選擇 STL 檔案", "", "STL Files (*.stl)")
        if not filepath:
            return

        self.reset_app_state()

        # --- 重要升級：讀取使用者設定的閾值 ---
        try:
            min_volume_threshold = float(self.min_volume_input.text())
        except ValueError:
            QMessageBox.warning(self, "輸入無效", "最小體積閾值不是一個有效的數字，將使用預設值 1e-6。")
            min_volume_threshold = 1e-6
            self.min_volume_input.setText("1e-6")

        try:
            self.scene_mesh = trimesh.load_mesh(filepath, process=False)
            self.particles = self.scene_mesh.split(only_watertight=False)
            
            if not self.particles:
                QMessageBox.warning(self, "錯誤", "無法在此檔案中偵測到任何獨立的顆粒。")
                return

            # 將閾值傳遞給處理函式
            skipped_count = self.process_all_particles(min_volume_threshold)
            
            if not self.particle_data:
                QMessageBox.warning(self, "處理完成", f"檔案中包含 {len(self.particles)} 個物件，但根據您設定的最小體積閾值 ({min_volume_threshold} nm³)，沒有一個是有效的3D顆粒。")
                return

            self.populate_global_plotter()
            self.populate_table()

            self.action_export.setEnabled(True)
            self.action_dist_vis.setEnabled(True)
            self.clear_detail_view()
            
            success_msg = f"成功載入並分析了 {len(self.particle_data)} 個有效顆粒。"
            if skipped_count > 0:
                success_msg += f"\n\n(已自動跳過 {skipped_count} 個體積過小或無效的物件)"
            QMessageBox.information(self, "成功", success_msg)

        except Exception as e:
            QMessageBox.critical(self, "載入失敗", f"讀取或處理檔案時發生錯誤：\n{str(e)}")
            self.reset_app_state()

    def process_all_particles(self, min_volume_threshold):
        self.particle_data = []
        skipped_count = 0
        
        for i, p_mesh in enumerate(self.particles):
            try:
                if len(p_mesh.vertices) < 4:
                    skipped_count += 1
                    continue

                if not p_mesh.is_watertight:
                    p_mesh.fill_holes()
                if p_mesh.volume < 0:
                    p_mesh.invert()

                # --- 重要升級：使用使用者定義的閾值 ---
                if p_mesh.volume < min_volume_threshold:
                    skipped_count += 1
                    continue

                V = p_mesh.volume
                As = p_mesh.area
                convex_hull = p_mesh.convex_hull
                V_ch = convex_hull.volume
                
                if V_ch < min_volume_threshold:
                    skipped_count += 1
                    continue

                obb_extents = p_mesh.bounding_box_oriented.extents
                a, b, c = sorted(obb_extents, reverse=True)

                sphericity = (np.pi**(1/3) * (6 * V)**(2/3)) / As if As > 1e-9 else 0
                convexity = V / V_ch
                denominator = 4 * np.pi * ((a + b + c) / 6)**2
                roundness = As / denominator if denominator > 1e-9 else 0

                self.particle_data.append({
                    "id": len(self.particle_data),
                    "mesh": p_mesh, "a": a, "b": b, "c": c,
                    "V": V, "V_ch": V_ch, "Area": As, "Sphericity": sphericity,
                    "Convexity": convexity, "Roundness": roundness,
                })
            except Exception:
                skipped_count += 1
                continue
        
        return skipped_count

    # ... (populate_global_plotter, on_pick, update_detail_view, etc. 函式與之前相同，為節省篇幅省略) ...
    def populate_global_plotter(self):
        self.plotter_global.clear()
        if not self.particle_data: return
        valid_meshes = [p['mesh'] for p in self.particle_data]
        display_mesh = trimesh.util.concatenate(valid_meshes)
        
        self_actor = self.plotter_global.add_mesh(display_mesh, color='lightgrey', style='surface', opacity=0.8)
        self_actor.name = 'full_scene'
        self.plotter_global.enable_mesh_picking(self.on_pick)
        self.plotter_global.reset_camera()

    def on_pick(self, *args):
        if not self.plotter_global.picked_point or not self.particle_data:
            return
            
        picked_point = self.plotter_global.picked_point
        centroids = np.array([p['mesh'].centroid for p in self.particle_data])
        distances = np.linalg.norm(centroids - picked_point, axis=1)
        picked_id = np.argmin(distances)

        if self.selected_particle_index != -1:
            prev_actor = self.plotter_global.actors.get(f"particle_highlight")
            if prev_actor: self.plotter_global.remove_actor(prev_actor)

        self.selected_particle_index = picked_id
        
        highlight_mesh = self.particle_data[picked_id]['mesh']
        self.plotter_global.add_mesh(highlight_mesh, color='cyan', name="particle_highlight", line_width=5)
        
        self.update_detail_view()
        self.table.selectRow(picked_id)

    def update_detail_view(self):
        if self.selected_particle_index == -1 or self.selected_particle_index >= len(self.particle_data):
            self.clear_detail_view()
            return
        
        self.plotter_detail.clear()
        data = self.particle_data[self.selected_particle_index]
        mesh = data["mesh"]
        
        self.plotter_detail.add_mesh(mesh, color="gold", name="main_particle")

        if self.cb_convex_hull.isChecked():
            self.plotter_detail.add_mesh(mesh.convex_hull, style='wireframe', color='red', line_width=2)

        if self.cb_ellipsoid.isChecked():
            try:
                obb = mesh.bounding_box_oriented
                ellipsoid = pv.Sphere()
                ellipsoid.points *= np.array(obb.extents) / 2.0
                ellipsoid.apply_transform(obb.transform)
                self.plotter_detail.add_mesh(ellipsoid, color='lightblue', opacity=0.5)
            except Exception: pass

        if self.cb_axes.isChecked():
            try:
                obb = mesh.bounding_box_oriented
                center = obb.primitive.center
                vectors = obb.primitive.vectors
                extents = obb.primitive.extents
                self.plotter_detail.add_arrows(center, vectors[0] * extents[0], mag=1.0, color='red')
                self.plotter_detail.add_arrows(center, vectors[1] * extents[1], mag=1.0, color='green')
                self.plotter_detail.add_arrows(center, vectors[2] * extents[2], mag=1.0, color='blue')
            except Exception: pass

        self.plotter_detail.reset_camera()

    def clear_detail_view(self):
        self.plotter_detail.clear()
        self.selected_particle_index = -1
        self.cb_ellipsoid.setChecked(False)
        self.cb_convex_hull.setChecked(False)
        self.cb_axes.setChecked(False)

    def populate_table(self):
        if not self.particle_data: return
        
        headers = ["顆粒ID", "主軸a(nm)", "中軸b(nm)", "短軸c(nm)", "體積V(nm³)", "凸包體積V_CH(nm³)", "表面積A(nm²)",
                   "圓度指數(IR)", "球形度(S)", "凸性(Cx)"]
        
        self.table.setRowCount(len(self.particle_data))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        
        for row, data in enumerate(self.particle_data):
            self.table.setItem(row, 0, QTableWidgetItem(f"{data['id']+1}/{len(self.particle_data)}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{data['a']:.3f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{data['b']:.3f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{data['c']:.3f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{data['V']:.3f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{data['V_ch']:.3f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{data['Area']:.3f}"))
            self.table.setItem(row, 7, QTableWidgetItem(f"{data['Roundness']:.3f}"))
            self.table.setItem(row, 8, QTableWidgetItem(f"{data['Sphericity']:.3f}"))
            self.table.setItem(row, 9, QTableWidgetItem(f"{data['Convexity']:.3f}"))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)

    def on_table_selection_changed(self, selected, deselected):
        if not selected.indexes(): return
        
        row = selected.indexes()[0].row()
        if self.selected_particle_index != row:
            if self.selected_particle_index != -1:
                prev_actor = self.plotter_global.actors.get(f"particle_highlight")
                if prev_actor: self.plotter_global.remove_actor(prev_actor)

            self.selected_particle_index = row
            highlight_mesh = self.particle_data[row]['mesh']
            self.plotter_global.add_mesh(highlight_mesh, color='cyan', name="particle_highlight", line_width=5)
            self.update_detail_view()

    def export_to_csv(self):
        if not self.particle_data:
            QMessageBox.warning(self, "無數據", "沒有可匯出的數據。")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(self, "匯出為 CSV", "", "CSV Files (*.csv)")
        if not filepath: return
            
        headers = [h for h in list(self.particle_data[0].keys()) if h != 'mesh']

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for data in self.particle_data:
                    row_to_write = {k: v for k, v in data.items() if k != 'mesh'}
                    writer.writerow(row_to_write)
            QMessageBox.information(self, "成功", f"數據已成功匯出至 {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", f"寫入檔案時發生錯誤：\n{str(e)}")

    def visualize_distribution(self):
        if len(self.particle_data) < 4:
            QMessageBox.warning(self, "數據不足", "需要至少4個顆粒才能進行空間分佈密度分析。")
            return

        try:
            centroids = np.array([p['mesh'].centroid for p in self.particle_data])
            kde = gaussian_kde(centroids.T)

            bounds = self.scene_mesh.bounds
            x_min, x_max = bounds[0]; y_min, y_max = bounds[1]; z_min, z_max = bounds[2]
            
            dim = 50
            xi, yi, zi = np.mgrid[x_min:x_max:dim*1j, y_min:y_max:dim*1j, z_min:z_max:dim*1j]
            coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
            
            density = kde(coords).reshape(xi.shape)

            grid = pv.StructuredGrid(xi, yi, zi)
            grid["density"] = density.ravel(order="F")

            self.plotter_global.clear()
            self.plotter_global.add_volume(grid, cmap="magma", opacity="linear", shade=True)
            self.plotter_global.add_text("顆粒空間分佈密度\n(顏色越亮密度越高)", font_size=12)
            self.plotter_global.reset_camera()
            
            QMessageBox.information(self, "分析完成", "已在左側視窗顯示空間分佈密度。\n請重新載入檔案以返回顆粒選擇模式。")

        except Exception as e:
            QMessageBox.critical(self, "分析失敗", f"計算空間分佈時發生錯誤：\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = ParticleAnalyzerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()