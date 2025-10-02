import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import glob
import os
from tqdm import tqdm
import warnings
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from mpl_toolkits.mplot3d import Axes3D

# 忽略警告
warnings.filterwarnings('ignore')

# 設定中文字體支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 使用完整的互動模式
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg後端支持完整互動

# 設定基本參數
PIXEL_SIZE = 0.65  # 每像素 0.65 μm

print("=== 分割孔隙影像的網路建模驗證程式 ===")
print(f"像素尺寸: {PIXEL_SIZE} μm/pixel")
print("使用已分割的孔隙影像進行快速驗證")

# 載入分割後的孔隙影像
print("\n=== 載入分割後的孔隙影像 ===")
tiff_folder = "./smallpore_0922_extract_2Dtiff"

if not os.path.exists(tiff_folder):
    print(f"❌ 找不到分割影像資料夾: {tiff_folder}")
    exit(1)

# 取得所有 view*.tif 檔案並排序
tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.view*.tif")))
print(f"找到 {len(tiff_files)} 個分割影像檔案")

if len(tiff_files) == 0:
    print("❌ 資料夾中沒有找到分割影像檔案")
    exit(1)

# 載入第一張影像來確定尺寸
first_image = imread(tiff_files[0])
height, width = first_image.shape
print(f"影像尺寸: {width} x {height} pixels")

# 載入所有分割影像
depth = len(tiff_files)
print(f"將載入 {depth} 張分割影像")

try:
    im_3d = np.zeros((depth, height, width), dtype=np.uint8)
    
    # 載入所有切片
    for i, file_path in enumerate(tqdm(tiff_files, desc="載入分割影像")):
        try:
            img = imread(file_path)
            # 確保影像是二值化的
            if img.max() > 1:
                img = (img > 127).astype(np.uint8)
            im_3d[i] = img
        except Exception as e:
            print(f"載入 {file_path} 失敗: {e}")
            continue
            
    print(f"✓ 成功載入分割影像: {im_3d.shape}")
    print(f"記憶體使用: {im_3d.nbytes / (1024**2):.1f} MB")
    
except MemoryError:
    print("❌ 記憶體不足")
    exit(1)

# 驗證和預處理
unique_values = np.unique(im_3d)
print(f"影像值範圍: {unique_values}")

if len(unique_values) > 2:
    print("⚠️  進行二值化處理...")
    im_3d = (im_3d > 0).astype(np.uint8)

# 檢查孔隙率
porosity = np.sum(im_3d) / im_3d.size
print(f"孔隙率: {porosity*100:.2f}%")

# 如果孔隙率太低，可能需要反轉
if porosity < 0.1:
    print("⚠️  孔隙率過低，反轉影像...")
    im_3d = 1 - im_3d
    porosity = np.sum(im_3d) / im_3d.size
    print(f"反轉後孔隙率: {porosity*100:.2f}%")

# 分割孔隙檢測和分析
print("\n=== 分割孔隙結構分析 ===")

# 1. 清理小雜訊
min_pore_size = 100  # 適中的最小孔隙尺寸
cleaned = remove_small_objects(im_3d.astype(bool), min_size=min_pore_size)
im_cleaned = cleaned.astype(np.uint8)

print(f"清理後孔隙率: {np.sum(im_cleaned) / im_cleaned.size * 100:.2f}%")

# 2. 距離變換找孔隙中心
distance_transform = ndimage.distance_transform_edt(im_cleaned)
print("✓ 完成距離變換")

# 3. 檢測孔隙中心（使用適中的參數）
min_distance = 8  # 適中的最小孔隙間距
local_maxima = ndimage.maximum_filter(distance_transform, size=min_distance) == distance_transform
local_maxima &= (distance_transform > 4)  # 適中的最小半徑閾值

# 4. 標記孔隙區域
markers = label(local_maxima)
from skimage.segmentation import watershed
segmented_pores = watershed(-distance_transform, markers, mask=im_cleaned)

# 5. 計算孔隙屬性
pore_regions = regionprops(segmented_pores, intensity_image=distance_transform)
print(f"✓ 檢測到 {len(pore_regions)} 個分割孔隙")

# 提取有效孔隙數據
pore_centers = []
pore_diameters = []
pore_volumes = []

for region in pore_regions:
    center = region.centroid
    max_radius = region.max_intensity
    diameter = 2 * max_radius * PIXEL_SIZE
    volume = region.area * (PIXEL_SIZE ** 3)
    
    # 較寬鬆的過濾條件以保留更多孔隙
    if diameter > 2.0 and volume > 20:
        pore_centers.append(center)
        pore_diameters.append(diameter)
        pore_volumes.append(volume)

pore_centers = np.array(pore_centers)
pore_diameters = np.array(pore_diameters)
pore_volumes = np.array(pore_volumes)

print(f"✓ 有效孔隙數量: {len(pore_centers)}")
if len(pore_centers) > 0:
    print(f"  直徑範圍: {pore_diameters.min():.2f} - {pore_diameters.max():.2f} μm")
    print(f"  平均直徑: {pore_diameters.mean():.2f} μm")

# 智能喉道建模（針對分割影像優化）
print("\n=== 喉道連接建模 ===")

throat_connections = []
throat_lengths = []
throat_diameters = []

if len(pore_centers) > 1:
    # 使用 KD 樹快速查找鄰近孔隙
    tree = cKDTree(pore_centers)
    
    max_neighbors = 4  # 減少連接數，更符合實際
    
    for i in range(len(pore_centers)):
        # 搜索範圍基於孔隙大小
        search_radius = pore_diameters[i] / PIXEL_SIZE * 2.5
        neighbors = tree.query_ball_point(pore_centers[i], search_radius)
        
        # 移除自己，只連接索引更大的點避免重複
        neighbors = [n for n in neighbors if n > i]
        
        # 限制鄰居數量並按距離排序
        if len(neighbors) > max_neighbors:
            neighbor_distances = [np.linalg.norm(pore_centers[i] - pore_centers[n]) for n in neighbors]
            sorted_indices = np.argsort(neighbor_distances)
            neighbors = [neighbors[idx] for idx in sorted_indices[:max_neighbors]]
        
        for j in neighbors:
            # 計算實際距離
            pixel_distance = np.linalg.norm(pore_centers[i] - pore_centers[j])
            actual_distance = pixel_distance * PIXEL_SIZE
            
            # 連接條件
            avg_diameter = (pore_diameters[i] + pore_diameters[j]) / 2
            min_distance = avg_diameter * 0.5  # 最小距離
            max_distance = avg_diameter * 4.0  # 最大距離
            
            if min_distance < actual_distance < max_distance:
                throat_connections.append([i, j])
                throat_lengths.append(actual_distance)
                
                # 喉道直徑為較小孔隙的 60%
                throat_diameter = min(pore_diameters[i], pore_diameters[j]) * 0.6
                throat_diameters.append(throat_diameter)

throat_connections = np.array(throat_connections)
throat_lengths = np.array(throat_lengths)
throat_diameters = np.array(throat_diameters)

print(f"✓ 建立 {len(throat_connections)} 個喉道連接")
if len(throat_connections) > 0:
    print(f"  平均喉道長度: {throat_lengths.mean():.2f} μm")
    print(f"  平均喉道直徑: {throat_diameters.mean():.2f} μm")

# 計算連接性
connectivity = np.zeros(len(pore_centers))
if len(throat_connections) > 0:
    for connection in throat_connections:
        connectivity[connection[0]] += 1
        connectivity[connection[1]] += 1

print(f"  平均連接數: {connectivity.mean():.2f}")

# 創建互動式孔隙網路可視化
print("\n=== 創建孔隙網路可視化 ===")

if len(pore_centers) > 0:
    # 轉換座標為實際尺寸
    coords_um = pore_centers * PIXEL_SIZE
    
    # 創建3D圖表（全屏互動模式）
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 啟用所有互動功能
    fig.canvas.toolbar_visible = True
    
    # 設定點大小範圍
    min_size = 30
    max_size = 800
    if len(pore_diameters) > 1:
        normalized_sizes = min_size + (max_size - min_size) * (pore_diameters - pore_diameters.min()) / (pore_diameters.max() - pore_diameters.min())
    else:
        normalized_sizes = np.full(len(pore_diameters), (min_size + max_size) / 2)
    
    # 繪製孔隙：點大小和顏色都表示直徑
    scatter = ax.scatter(coords_um[:, 2], coords_um[:, 1], coords_um[:, 0],
                        s=normalized_sizes,
                        c=pore_diameters,
                        cmap='plasma',  # 使用更鮮豔的顏色
                        alpha=0.8,
                        edgecolors='black',
                        linewidth=0.8)
    
    # 繪製喉道連接
    if len(throat_connections) > 0:
        for k, (i, j) in enumerate(throat_connections):
            if i < len(coords_um) and j < len(coords_um):
                # 線條粗細反映喉道直徑
                line_width = max(0.5, throat_diameters[k] / throat_diameters.max() * 4)
                
                ax.plot([coords_um[i, 2], coords_um[j, 2]],
                       [coords_um[i, 1], coords_um[j, 1]], 
                       [coords_um[i, 0], coords_um[j, 0]],
                       color='gray', linewidth=line_width, alpha=0.7)
    
    # 設定圖表屬性
    ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (μm)', fontsize=12, fontweight='bold')
    
    # 設定標題
    title = f'分割孔隙網路模型驗證\n孔隙數: {len(pore_centers)} | 喉道數: {len(throat_connections)} | 孔隙率: {porosity*100:.1f}%'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加顏色條
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30, pad=0.1)
    cbar.set_label('孔隙直徑 (μm)', fontsize=12, fontweight='bold')
    
    # 添加統計信息
    if len(pore_diameters) > 0:
        stats_text = f"""網路統計:
• 孔隙數量: {len(pore_centers)}
• 喉道數量: {len(throat_connections)}
• 平均孔隙直徑: {pore_diameters.mean():.1f} μm
• 直徑範圍: {pore_diameters.min():.1f} - {pore_diameters.max():.1f} μm
• 平均連接數: {connectivity.mean():.1f}
• 孔隙率: {porosity*100:.1f}%"""
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                 verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 設定更好的視角
    ax.view_init(elev=25, azim=45)
    
    # 改善圖表布局
    plt.tight_layout()
    
    # 啟用完整互動模式
    plt.ion()
    plt.show(block=False)
    
    print("✓ 互動式 3D 可視化已顯示")
    print("  🖱️  滑鼠操作：")
    print("    • 左鍵拖拉：旋轉視角")
    print("    • 右鍵拖拉：縮放")
    print("    • 中鍵拖拉：平移")
    print("    • 滾輪：快速縮放")
    print("  🎨 可視化說明：")
    print("    • 點大小和顏色都表示孔隙直徑")
    print("    • 線條粗細表示喉道直徑")
    print("    • 可使用工具列按鈕進行更多操作")

else:
    print("❌ 沒有檢測到有效孔隙，無法進行可視化")

# 輸出驗證結果
print(f"\n=== 驗證結果摘要 ===")
print(f"📊 數據統計:")
print(f"  影像尺寸: {im_3d.shape}")
print(f"  實際尺寸: {im_3d.shape[2]*PIXEL_SIZE:.1f} × {im_3d.shape[1]*PIXEL_SIZE:.1f} × {im_3d.shape[0]*PIXEL_SIZE:.1f} μm³")
print(f"  孔隙率: {porosity*100:.2f}%")

if len(pore_centers) > 0:
    print(f"\n🔵 孔隙分析:")
    print(f"  檢測數量: {len(pore_centers)}")
    print(f"  直徑統計: {pore_diameters.min():.2f} - {pore_diameters.max():.2f} μm (平均: {pore_diameters.mean():.2f} μm)")
    print(f"  體積統計: {pore_volumes.min():.1f} - {pore_volumes.max():.1f} μm³")

if len(throat_connections) > 0:
    print(f"\n🔗 喉道分析:")
    print(f"  連接數量: {len(throat_connections)}")
    print(f"  長度統計: {throat_lengths.min():.2f} - {throat_lengths.max():.2f} μm (平均: {throat_lengths.mean():.2f} μm)")
    print(f"  直徑統計: {throat_diameters.min():.2f} - {throat_diameters.max():.2f} μm (平均: {throat_diameters.mean():.2f} μm)")

print(f"\n🌐 網路連接性:")
if len(connectivity) > 0:
    print(f"  平均連接數: {connectivity.mean():.2f}")
    print(f"  連接數範圍: {connectivity.min():.0f} - {connectivity.max():.0f}")
    print(f"  連接密度: {len(throat_connections)/(len(pore_centers) if len(pore_centers) > 0 else 1):.2f}")

# 驗證評估
print(f"\n✅ 演算法驗證:")
if len(pore_centers) > 5:
    print("  ✓ 孔隙檢測: 成功檢測到多個孔隙")
else:
    print("  ⚠️  孔隙檢測: 檢測到的孔隙數量較少")

if len(throat_connections) > 0:
    avg_connectivity = connectivity.mean()
    if 2 <= avg_connectivity <= 8:
        print("  ✓ 喉道建模: 連接性合理")
    elif avg_connectivity < 2:
        print("  ⚠️  喉道建模: 連接性偏低")
    else:
        print("  ⚠️  喉道建模: 連接性偏高")
else:
    print("  ❌ 喉道建模: 未建立連接")

if 5 <= porosity*100 <= 50:
    print("  ✓ 孔隙率: 數值合理")
else:
    print("  ⚠️  孔隙率: 可能需要調整")

print(f"\n=== 分割影像驗證完成 ===")
print("如果結果合理，可以進行完整影像的複雜分析")

# 等待用戶確認
input("\n按 Enter 鍵關閉程式...")
plt.close('all')