import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from skimage import measure
import trimesh
import panel as pn

# 明確載入 PyVista 擴展
pn.extension('pyvista')

# 設定Pyvista主題
pv.set_plot_theme("document")

# 載入3D模型
mesh = pv.read(r"C:\Users\User\Desktop\sp051-12\SP051 STL\3D_SP051_120.stl")

# 基本幾何特性計算
volume = mesh.volume
surface_area = mesh.area
ratio = surface_area/volume
print(f"體積:{volume:.4f} 立方微米")
print(f"表面積:{surface_area:.4f} 平方微米")
print(f"表面積/體積比:{ratio:.4f}")

# 孔隙率分析
def analyze_porosity(mesh):
    bounds = mesh.bounds
    total_volume = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])
    solid_volume = mesh.volume
    porosity = 1 - (solid_volume / total_volume)
    print(f"孔隙率: {porosity:.4f} ({porosity*100:.2f}%)")
    return porosity

# 厚度分析（基於曲率）
def analyze_thickness(mesh):
    # 計算網格的曲率
    mesh.compute_normals(inplace=True)
    curv = mesh.curvature()
    
    # 創建視圖來顯示曲率
    p = pv.Plotter(notebook=True)
    p.add_mesh(mesh, scalars=curv, cmap='viridis', show_edges=False)
    p.add_scalar_bar(title='曲率 (估計厚度)')
    p.add_axes()
    
    # 計算並顯示統計信息
    print(f"最大曲率: {max(curv):.4f}")
    print(f"最小曲率: {min(curv):.4f}")
    print(f"平均曲率: {sum(curv)/len(curv):.4f}")
    
    return p

# 連通性分析
def analyze_connectivity(mesh):
    # 獲取連通區域
    try:
        labeled = mesh.connectivity(largest=False)
        n_regions = labeled.n_values
        
        # 創建視圖來顯示不同連通區域
        p = pv.Plotter(notebook=True)
        p.add_mesh(labeled, cmap='tab20', show_edges=False)
        p.add_scalar_bar(title='連通區域ID')
        p.add_axes()
        
        print(f"連通區域數量: {n_regions}")
        
        # 計算每個連通區域的體積
        volumes = []
        for i in range(n_regions):
            region = labeled.threshold([i, i])
            volumes.append(region.volume)
        
        if volumes:
            print(f"最大連通區域體積: {max(volumes):.4f}")
            print(f"最小連通區域體積: {min(volumes):.4f}")
        
        return p
    except Exception as e:
        print(f"連通性分析出錯: {e}")
        p = pv.Plotter(notebook=True)
        p.add_mesh(mesh, color='red')
        p.add_text("連通性分析失敗")
        return p

# 基本視覺化
def basic_visualization(mesh):
    p = pv.Plotter(notebook=True)
    p.add_mesh(mesh, color='lightgray', show_edges=False)
    p.add_axes()
    
    # 添加文字信息
    text = f"體積: {volume:.4f}\n表面積: {surface_area:.4f}\n比率: {ratio:.4f}"
    p.add_text(text, position='upper_left')
    
    return p

# 截面視覺化
def slice_visualization(mesh):
    p = pv.Plotter(notebook=True)
    
    # 初始顯示完整模型
    actor = p.add_mesh(mesh, color='lightgray', show_edges=False)
    p.add_axes()
    
    # 截面位置初始值
    x_pos = (mesh.bounds[0] + mesh.bounds[1]) / 2
    
    # 更新截面的函數
    def update_slice(x):
        # 移除現有網格
        p.remove_actor(actor)
        
        # 創建新的截面
        slice_x = mesh.slice(normal='x', origin=(x, 0, 0))
        
        # 添加截面和半透明的原始模型
        p.add_mesh(slice_x, color='red')
        p.add_mesh(mesh, color='lightgray', opacity=0.3)
        
        p.render()
    
    # 添加滑塊控制
    slider = pn.widgets.FloatSlider(
        name='X截面位置',
        start=mesh.bounds[0],
        end=mesh.bounds[1],
        value=x_pos,
        step=(mesh.bounds[1]-mesh.bounds[0])/100
    )
    
    # 將滑塊與更新函數連接
    slider.param.watch(lambda event: update_slice(event.new), 'value')
    
    return p, slider

# 執行各種分析
porosity = analyze_porosity(mesh)

# 創建各種視覺化
basic_view = pn.pane.PyVista(basic_visualization(mesh), height=500)
thickness_view = pn.pane.PyVista(analyze_thickness(mesh), height=500)
connectivity_view = pn.pane.PyVista(analyze_connectivity(mesh), height=500)
slice_view, slice_slider = slice_visualization(mesh)
slice_panel = pn.Column(slice_slider, pn.pane.PyVista(slice_view, height=500))

# 創建選項卡式界面
tabs = pn.Tabs(
    ('基本視覺化', basic_view),
    ('厚度分析', thickness_view),
    ('連通性分析', connectivity_view),
    ('截面視圖', slice_panel)
)

# 創建儀表板標題和說明
title = pn.pane.Markdown('# 微觀結構分析儀表板')
description = pn.pane.Markdown(f"""
## 基本幾何特性
- **體積**: {volume:.4f} 立方微米
- **表面積**: {surface_area:.4f} 平方微米
- **表面積/體積比**: {ratio:.4f}
- **孔隙率**: {porosity*100:.2f}%
""")

# 組合儀表板
dashboard = pn.Column(
    title,
    description,
    tabs,
    width=800
)

# 顯示儀表板
dashboard.servable('微觀結構分析')
# 或使用 dashboard.show() 在筆記本中顯示
