import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import glob
import os
from tqdm import tqdm
import warnings
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# 設定 matplotlib 為非互動模式，避免程式卡住
import matplotlib
matplotlib.use('Agg')  # 非互動後端
plt.ioff()  # 關閉互動模式

# 忽略警告
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.family'] = 'DejaVu Sans'

# 設定基本參數
PIXEL_SIZE = 0.65  # 每像素 0.65 μm

print("=== 簡化版孔隙網路分析程式 ===")
print(f"像素尺寸: {PIXEL_SIZE} μm/pixel")

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
max_slices = 50  # 進一步限制切片數量
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

# 簡化的孔隙分析
print("\n=== 執行簡化孔隙分析 ===")
print("⏳ 正在分析孔隙結構...")

try:
    # 移除小物件（雜訊）
    min_size = 50  # 最小孔隙大小
    cleaned = remove_small_objects(im_3d.astype(bool), min_size=min_size)
    im_3d = cleaned.astype(np.uint8)
    
    # 標記連通組件
    labeled_pores = label(im_3d, connectivity=3)  # 3D 連通性
    
    # 計算孔隙屬性
    pore_props = regionprops(labeled_pores)
    
    num_pores = len(pore_props)
    print(f"✓ 發現 {num_pores} 個孔隙")
    
    # 計算等效直徑
    pore_volumes = [prop.area for prop in pore_props]  # 在 3D 中，area 實際上是體積
    pore_diameters = []
    
    for volume in pore_volumes:
        # 等效球體直徑
        diameter = 2 * ((3 * volume) / (4 * np.pi)) ** (1/3) * PIXEL_SIZE
        pore_diameters.append(diameter)
    
    pore_diameters = np.array(pore_diameters)
    
    print(f"  平均孔隙直徑: {pore_diameters.mean():.2f} μm")
    print(f"  孔隙直徑範圍: {pore_diameters.min():.2f} - {pore_diameters.max():.2f} μm")
    
except Exception as e:
    print(f"❌ 孔隙分析失敗: {e}")
    exit(1)

print("✓ 簡化分析完成！")

# 分析結果
print("\n=== 分析結果 ===")
print(f"影像尺寸: {im_3d.shape}")
print(f"實際尺寸: {im_3d.shape[0]*PIXEL_SIZE:.1f} x {im_3d.shape[1]*PIXEL_SIZE:.1f} x {im_3d.shape[2]*PIXEL_SIZE:.1f} μm³")
print(f"孔隙數量: {num_pores}")
print(f"孔隙率: {porosity*100:.2f}%")

# 計算距離變換來估計孔隙大小分佈
print("\n=== 距離變換分析 ===")
distance_transform = ndimage.distance_transform_edt(im_3d)
local_maxima = (distance_transform > 0.5 * distance_transform.max())
pore_centers = np.where(local_maxima)

# 計算基於距離變換的直徑
dt_diameters = 2 * distance_transform[local_maxima] * PIXEL_SIZE
print(f"基於距離變換的孔隙中心數量: {len(dt_diameters)}")

if len(dt_diameters) > 0:
    print(f"距離變換直徑範圍: {dt_diameters.min():.2f} - {dt_diameters.max():.2f} μm")
    print(f"平均直徑: {dt_diameters.mean():.2f} μm")

# 繪製分析圖表
print("\n=== 生成分析圖表 ===")
plt.figure(figsize=(15, 10))

# 圖1: 孔隙直徑分佈
plt.subplot(2, 3, 1)
if len(pore_diameters) > 0:
    plt.hist(pore_diameters, bins=min(30, len(pore_diameters)), edgecolor='k', alpha=0.7, color='skyblue')
    plt.xlabel('孔隙直徑 (μm)', fontsize=12)
    plt.ylabel('數量', fontsize=12)
    plt.title('孔隙尺寸分佈\n(基於連通組件)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    mean_d = pore_diameters.mean()
    median_d = np.median(pore_diameters)
    plt.axvline(mean_d, color='red', linestyle='--', label=f'平均: {mean_d:.1f} μm')
    plt.axvline(median_d, color='green', linestyle='--', label=f'中位數: {median_d:.1f} μm')
    plt.legend()

# 圖2: 距離變換直徑分佈
plt.subplot(2, 3, 2)
if len(dt_diameters) > 0:
    plt.hist(dt_diameters, bins=min(50, len(dt_diameters)), edgecolor='k', alpha=0.7, color='lightcoral')
    plt.xlabel('孔隙直徑 (μm)', fontsize=12)
    plt.ylabel('數量', fontsize=12)
    plt.title('孔隙尺寸分佈\n(基於距離變換)', fontsize=12)
    plt.grid(True, alpha=0.3)

# 圖3: 孔隙體積分佈
plt.subplot(2, 3, 3)
if len(pore_volumes) > 0:
    volumes_um3 = np.array(pore_volumes) * (PIXEL_SIZE ** 3)
    plt.hist(volumes_um3, bins=min(30, len(volumes_um3)), edgecolor='k', alpha=0.7, color='lightgreen')
    plt.xlabel('孔隙體積 (μm³)', fontsize=12)
    plt.ylabel('數量', fontsize=12)
    plt.title('孔隙體積分佈', fontsize=12)
    plt.grid(True, alpha=0.3)

# 圖4: 2D 切片示例
plt.subplot(2, 3, 4)
middle_slice = im_3d.shape[0] // 2
plt.imshow(im_3d[middle_slice], cmap='gray')
plt.title(f'2D 切片示例\n(第 {middle_slice} 層)', fontsize=12)
plt.axis('off')

# 圖5: 距離變換示例
plt.subplot(2, 3, 5)
dt_slice = distance_transform[middle_slice]
plt.imshow(dt_slice, cmap='viridis')
plt.title('距離變換\n(中間切片)', fontsize=12)
plt.colorbar(label='距離 (pixels)')
plt.axis('off')

# 圖6: 統計摘要
plt.subplot(2, 3, 6)
plt.axis('off')
stats_text = f"""統計摘要:

影像尺寸: {im_3d.shape}
實際尺寸: 
{im_3d.shape[0]*PIXEL_SIZE:.1f} × 
{im_3d.shape[1]*PIXEL_SIZE:.1f} × 
{im_3d.shape[2]*PIXEL_SIZE:.1f} μm³

孔隙率: {porosity*100:.2f}%
孔隙數量: {num_pores}

孔隙直徑統計:
平均: {pore_diameters.mean():.2f} μm
中位數: {np.median(pore_diameters):.2f} μm
標準差: {pore_diameters.std():.2f} μm
範圍: {pore_diameters.min():.2f} - {pore_diameters.max():.2f} μm
"""
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()

# 保存圖表
output_file = 'simplified_pore_analysis_results.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ 分析圖表已保存至: {output_file}")
plt.close('all')

# 顯示最終統計資訊
print(f"\n=== 最終統計資訊 ===")
print(f"✓ 孔隙直徑統計:")
print(f"  最小值: {pore_diameters.min():.2f} μm")
print(f"  最大值: {pore_diameters.max():.2f} μm")
print(f"  平均值: {pore_diameters.mean():.2f} μm")
print(f"  中位數: {np.median(pore_diameters):.2f} μm")
print(f"  標準差: {pore_diameters.std():.2f} μm")

if len(dt_diameters) > 0:
    print(f"\n✓ 距離變換統計:")
    print(f"  檢測到的孔隙中心: {len(dt_diameters)}")
    print(f"  平均直徑: {dt_diameters.mean():.2f} μm")
    print(f"  直徑範圍: {dt_diameters.min():.2f} - {dt_diameters.max():.2f} μm")

total_volume = im_3d.shape[0] * im_3d.shape[1] * im_3d.shape[2] * (PIXEL_SIZE ** 3)
pore_volume = np.sum(im_3d) * (PIXEL_SIZE ** 3)

print(f"\n✓ 體積統計:")
print(f"  總體積: {total_volume:.2f} μm³")
print(f"  孔隙體積: {pore_volume:.2f} μm³")
print(f"  孔隙率: {porosity*100:.2f}%")

print(f"\n=== 簡化分析完成 ===")
print(f"✓ 所有結果已保存為圖片檔案")
print(f"✓ 程式執行完畢")
print(f"  注意: 這是簡化版分析，如需完整的網路分析請安裝 PoreSpy 套件")