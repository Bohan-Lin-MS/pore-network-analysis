import porespy as ps
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from skimage.io import imread
import glob
import os
from tqdm import tqdm
import warnings
import psutil

# 設定 matplotlib 為非互動模式，避免程式卡住
import matplotlib
matplotlib.use('Agg')  # 非互動後端
plt.ioff()  # 關閉互動模式

# 檢查系統資源
def check_system_resources():
    """檢查系統記憶體使用情況"""
    memory = psutil.virtual_memory()
    print(f"系統記憶體: {memory.total / (1024**3):.1f} GB")
    print(f"可用記憶體: {memory.available / (1024**3):.1f} GB")
    print(f"記憶體使用率: {memory.percent:.1f}%")
    return memory.available / (1024**3) > 2.0  # 需要至少 2GB 可用記憶體

# 嘗試導入 CuPy (GPU 加速)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU 模式: CuPy 可用")
    
    # 檢查 GPU 記憶體
    mempool = cp.get_default_memory_pool()
    print(f"GPU 記憶體池大小: {mempool.used_bytes() / (1024**2):.1f} MB")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CPU 模式: CuPy 不可用，將使用 NumPy")

# 檢查系統資源
if not check_system_resources():
    print("⚠️  警告: 系統記憶體不足，可能影響性能")

# 忽略警告
warnings.filterwarnings('ignore')

# 設定中文字體
try:
    # 嘗試設定中文字體
    font_paths = [
        'C:/Windows/Fonts/msjh.ttc',  # 微軟正黑體
        'C:/Windows/Fonts/simhei.ttf',  # 黑體
        'C:/Windows/Fonts/simsun.ttc',  # 宋體
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"✓ 設定中文字體: {font_prop.get_name()}")
            break
    else:
        print("⚠️  未找到中文字體，使用預設字體")
        plt.rcParams['font.family'] = 'DejaVu Sans'
except Exception as e:
    print(f"⚠️  字體設定失敗: {e}")

# 設定基本參數
PIXEL_SIZE = 0.65  # 每像素 0.65 μm
SIGMA = 0.4
NOISE_THRESHOLD = 0.2

print("=== 孔隙網路分析程式 (GPU 加速版) ===")
print(f"像素尺寸: {PIXEL_SIZE} μm/pixel")
print(f"SNOW 算法參數: sigma={SIGMA}")

# 載入 3D 影像數據
print("\n=== 載入 3D 影像數據 ===")
tiff_folder = "./smallpore_0921_2Dtiff"

if not os.path.exists(tiff_folder):
    print(f"❌ 找不到資料夾: {tiff_folder}")
    exit(1)

# 取得所有 TIFF 檔案並排序
tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
print(f"找到 {len(tiff_files)} 個 TIFF 檔案")

if len(tiff_files) == 0:
    print("❌ 資料夾中沒有找到 TIFF 檔案")
    exit(1)

print("正在載入影像...")

# 載入第一張影像來確定尺寸
first_image = imread(tiff_files[0])
height, width = first_image.shape
print(f"影像尺寸: {width} x {height} pixels")

# 限制載入的切片數量以節省記憶體
max_slices = 100  # 限制最多載入 100 張切片
actual_files = tiff_files[:max_slices] if len(tiff_files) > max_slices else tiff_files

# 初始化 3D 陣列
depth = len(actual_files)
print(f"將載入 {depth} 張切片")

try:
    # 使用適當的數據類型
    im_3d = np.zeros((depth, height, width), dtype=np.uint8)
    
    # 載入所有切片
    for i, file_path in enumerate(tqdm(actual_files, desc="載入中")):
        try:
            img = imread(file_path)
            # 確保影像是二值化的
            if img.max() > 1:
                img = (img > 127).astype(np.uint8)  # 二值化
            im_3d[i] = img
        except Exception as e:
            print(f"載入 {file_path} 失敗: {e}")
            continue
            
    print(f"✓ 成功載入 3D 影像: {im_3d.shape}")
    print(f"數據類型: {im_3d.dtype}")
    print(f"記憶體使用: {im_3d.nbytes / (1024**2):.1f} MB")
    
except MemoryError:
    print("❌ 記憶體不足，請減少影像數量或尺寸")
    exit(1)

# 驗證數據
unique_values = np.unique(im_3d)
print(f"影像值範圍: {unique_values}")

if len(unique_values) > 2:
    print("⚠️  影像不是純二值，進行二值化處理...")
    im_3d = (im_3d > 0).astype(np.uint8)
    print(f"二值化後的值: {np.unique(im_3d)}")

# 確保孔隙為 1，固體為 0
if im_3d.sum() < im_3d.size * 0.1:
    print("⚠️  反轉影像 (孔隙/固體)")
    im_3d = 1 - im_3d

# 計算基本孔隙率
porosity = np.sum(im_3d) / im_3d.size
print(f"初始孔隙率: {porosity*100:.2f}%")

if porosity < 0.05 or porosity > 0.95:
    print("⚠️  異常的孔隙率值，請檢查影像數據")

# 執行 SNOW 算法
print("\n=== 執行 SNOW 算法 ===")
print("⏳ 正在分析孔隙網路...")

try:
    # 根據是否有 GPU 選擇不同的處理方式
    if GPU_AVAILABLE and im_3d.size < 50000000:  # 限制 GPU 使用的數據大小
        print("使用 GPU 加速...")
        # 將數據移到 GPU
        im_gpu = cp.asarray(im_3d)
        
        # 在 GPU 上執行 SNOW
        snow_output = ps.networks.snow2(
            im_gpu,
            voxel_size=PIXEL_SIZE,
            sigma=SIGMA,
            return_all=True
        )
        
        # 將結果移回 CPU
        for key, value in snow_output.items():
            if hasattr(value, 'get'):  # CuPy 陣列
                snow_output[key] = value.get()
                
        print("✓ GPU 處理完成")
        
    else:
        print("使用 CPU 處理...")
        snow_output = ps.networks.snow2(
            im_3d,
            voxel_size=PIXEL_SIZE,
            sigma=SIGMA,
            return_all=True
        )
        print("✓ CPU 處理完成")
        
except Exception as e:
    print(f"❌ SNOW 算法執行失敗: {e}")
    print("嘗試使用基本參數重新執行...")
    
    try:
        # 使用更保守的參數
        snow_output = ps.networks.snow2(
            im_3d,
            voxel_size=PIXEL_SIZE,
            sigma=0.3,  # 較小的 sigma
            return_all=True
        )
        print("✓ 重新執行成功")
    except Exception as e2:
        print(f"❌ 重新執行也失敗: {e2}")
        exit(1)

# 提取網路數據
print("\n=== 提取網路數據 ===")
try:
    pore_coords = snow_output['network']['pore.coords']
    throat_conns = snow_output['network']['throat.conns']
    
    print(f"孔隙數量: {len(pore_coords)}")
    print(f"喉道數量: {len(throat_conns)}")
    
    # 確保有孔隙直徑數據
    if 'pore.diameter' in snow_output:
        pore_diameters_um = snow_output['pore.diameter'] * PIXEL_SIZE
    else:
        # 如果沒有直徑數據，使用等效球體直徑
        pore_volumes = snow_output.get('pore.volume', np.ones(len(pore_coords)))
        pore_diameters_um = 2 * ((3 * pore_volumes) / (4 * np.pi)) ** (1/3)
        print("⚠️  使用估算的孔隙直徑")
    
    # 喉道特徵
    if 'throat.length' in snow_output:
        throat_lengths_um = snow_output['throat.length'] * PIXEL_SIZE
    else:
        # 計算喉道長度（兩個孔隙中心的距離）
        throat_lengths_um = np.array([
            np.linalg.norm(pore_coords[conn[1]] - pore_coords[conn[0]]) * PIXEL_SIZE
            for conn in throat_conns
        ])
        print("⚠️  使用估算的喉道長度")
    
    if 'throat.diameter' in snow_output:
        throat_diameters_um = snow_output['throat.diameter'] * PIXEL_SIZE
    else:
        # 估算喉道直徑
        throat_diameters_um = np.full(len(throat_conns), pore_diameters_um.mean() * 0.5)
        print("⚠️  使用估算的喉道直徑")
        
except Exception as e:
    print(f"❌ 數據提取失敗: {e}")
    exit(1)

print("✓ 網路數據提取完成！")

# 建立 OpenPNM 網路
print("\n=== 建立 OpenPNM 網路 ===")
try:
    pn = op.io.network_from_porespy(snow_output)
    print("✓ 成功建立網路")
except:
    pn = op.network.Network(coords=pore_coords, conns=throat_conns)

# 分析結果
print("\n=== 分析結果 ===")
print(f"影像尺寸: {im_3d.shape}")
print(f"實際尺寸: {im_3d.shape[0]*PIXEL_SIZE:.1f} x {im_3d.shape[1]*PIXEL_SIZE:.1f} x {im_3d.shape[2]*PIXEL_SIZE:.1f} μm³")
print(f"孔隙數量: {pn.Np}")
print(f"喉道數量: {pn.Nt}")
print(f"平均連接數: {pn.Nt * 2 / pn.Np:.2f}")

porosity = np.sum(im_3d) / im_3d.size
print(f"孔隙率: {porosity*100:.2f}%")

# 繪製分析圖表
plt.figure(figsize=(14, 6))

# 圖1: 孔隙直徑分佈
plt.subplot(1, 2, 1)
plt.hist(pore_diameters_um, bins=50, edgecolor='k', alpha=0.7, color='skyblue')
plt.xlabel('孔隙直徑 (μm)', fontsize=12)
plt.ylabel('數量', fontsize=12)
plt.title('孔隙尺寸分佈', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

mean_d = pore_diameters_um.mean()
median_d = np.median(pore_diameters_um)
plt.axvline(mean_d, color='red', linestyle='--', label=f'平均值: {mean_d:.1f} μm')
plt.axvline(median_d, color='green', linestyle='--', label=f'中位數: {median_d:.1f} μm')
plt.legend()

# 圖2: 喉道特徵散點圖
plt.subplot(1, 2, 2)
if len(throat_lengths_um) > 0 and len(throat_diameters_um) > 0:
    # 調試信息
    print(f"  喉道長度陣列形狀: {throat_lengths_um.shape}")
    print(f"  喉道直徑陣列形狀: {throat_diameters_um.shape}")
    
    # 確保陣列是一維的
    throat_lengths_flat = np.array(throat_lengths_um).flatten()
    throat_diameters_flat = np.array(throat_diameters_um).flatten()
    
    # 確保兩個陣列長度相同
    min_len = min(len(throat_lengths_flat), len(throat_diameters_flat))
    if min_len > 0:
        lengths_plot = throat_lengths_flat[:min_len]
        diameters_plot = throat_diameters_flat[:min_len]
        
        plt.scatter(lengths_plot, diameters_plot, alpha=0.6, s=10)
        plt.xlabel('喉道長度 (μm)', fontsize=12)
        plt.ylabel('喉道直徑 (μm)', fontsize=12)
        plt.title('喉道特徵關係', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        print(f"  使用了 {min_len} 個喉道數據點")
    else:
        plt.text(0.5, 0.5, '無可用的喉道數據', ha='center', va='center', transform=plt.gca().transAxes)
else:
    plt.text(0.5, 0.5, '無喉道數據', ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()

# 保存圖表而不是顯示（避免阻塞）
output_file = 'pore_analysis_results.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ 分析圖表已保存至: {output_file}")
plt.close('all')  # 關閉圖表釋放記憶體

# 3D 網路可視化 (可選)
print("\n=== 3D 網路可視化 ===")

# 設定是否要進行 3D 視覺化（大型數據集會很慢）
skip_3d = True  # 設為 False 如果你想要 3D 視覺化

if skip_3d:
    print("⚠️  跳過 3D 視覺化以提高性能")
    print("   如需 3D 視覺化，請將程式中的 skip_3d 設為 False")
    print("   大型數據集的 3D 視覺化可能需要很長時間")
else:
    print("⏳ 正在創建 3D 視覺化...")
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # 大幅減少數據量以提高性能
        max_pores = 50   # 只顯示少量數據點
        
        # 隨機選擇少量孔隙進行展示
        if len(pore_coords) > max_pores:
            indices = np.random.choice(len(pore_coords), max_pores, replace=False)
            coords_sample = pore_coords[indices] * PIXEL_SIZE
            diameters_sample = pore_diameters_um[indices]
        else:
            coords_sample = pore_coords * PIXEL_SIZE
            diameters_sample = pore_diameters_um
        
        # 創建簡單的 3D 散點圖
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(coords_sample[:, 0], coords_sample[:, 1], coords_sample[:, 2],
                           s=diameters_sample*5, c=diameters_sample,
                           cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'孔隙網路 3D 視覺化\n(顯示 {len(coords_sample)} 個孔隙)')
        
        plt.colorbar(scatter, label='孔隙直徑 (μm)')
        
        # 保存圖表而不是顯示
        output_file_3d = 'pore_network_3d.png'
        plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
        print(f"  ✓ 3D 視覺化已保存至: {output_file_3d}")
        plt.close()
        
    except Exception as e:
        print(f"  ✗ 3D 視覺化失敗: {e}")
        print("     這不影響主要分析結果")

# 顯示統計資訊
print(f"\n=== 最終統計資訊 ===")
print(f"孔隙直徑統計:")
print(f"  最小值: {pore_diameters_um.min():.2f} μm")
print(f"  最大值: {pore_diameters_um.max():.2f} μm")
print(f"  平均值: {mean_d:.2f} μm")
print(f"  中位數: {median_d:.2f} μm")
print(f"  標準差: {pore_diameters_um.std():.2f} μm")

if len(throat_lengths_um) > 0:
    print(f"\n喉道特徵統計:")
    print(f"  長度範圍: {throat_lengths_um.min():.2f} - {throat_lengths_um.max():.2f} μm")
    print(f"  平均長度: {throat_lengths_um.mean():.2f} μm")
    print(f"  直徑範圍: {throat_diameters_um.min():.2f} - {throat_diameters_um.max():.2f} μm")
    print(f"  平均直徑: {throat_diameters_um.mean():.2f} μm")

# 網路連接性分析
connectivity = []
for i in range(len(pore_coords)):
    connections = np.sum((throat_conns == i).any(axis=1))
    connectivity.append(connections)

connectivity = np.array(connectivity)
print(f"\n網路連接性統計:")
print(f"  平均連接數: {connectivity.mean():.2f}")
print(f"  最大連接數: {connectivity.max()}")
print(f"  連接數標準差: {connectivity.std():.2f}")

print(f"\n=== 分析完成 ===")
print(f"✓ 所有結果已保存為圖片檔案")
print(f"✓ 記憶體使用已優化")
print(f"✓ 程式執行完畢")