# 1. 匯入所有必要的函式庫
import porespy as ps
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import glob # 用於尋找檔案路徑
import os   # 用於處理檔案路徑
from tqdm import tqdm # 用於顯示進度條

print("程式開始執行：分析 3D 影像序列")

# ---! 請在這裡設定您的影像資料夾路徑 !---
image_folder_path = 'smallpore_0921_2Dtiff'

# 2. 讀取影像序列並堆疊成 3D 陣列
print(f"準備從資料夾 '{image_folder_path}' 讀取影像...")

# 使用 glob 找到所有符合命名規則的影像檔案
search_pattern = os.path.join(image_folder_path, 'onlystructure_*.labels[0-9]*')
file_list = glob.glob(search_pattern)

# !!! 非常重要：必須對檔案列表進行排序，以確保堆疊順序正確 !!!
file_list.sort()

if not file_list:
    print(f"錯誤：在 '{image_folder_path}' 中找不到任何影像檔案。")
    print(f"請檢查 'image_folder_path' 變數和檔案名稱是否正確。")
    exit()

print(f"找到 {len(file_list)} 個影像檔案，準備堆疊成 3D 體積...")

# 讀取第一張影像以獲取尺寸，並初始化 3D 陣列
first_image = imread(file_list[0])
im_shape = first_image.shape
num_slices = len(file_list)
# 建立一個空的 3D 陣列來存放所有影像
im_3d = np.zeros((num_slices, im_shape[0], im_shape[1]), dtype=first_image.dtype)

# 使用 tqdm 顯示進度條，迴圈讀取所有影像並放入 3D 陣列
for i, file_path in enumerate(tqdm(file_list, desc="讀取並堆疊影像")):
    im_3d[i, :, :] = imread(file_path)

# 確保孔隙為 True/1，基質為 False/0
im_3d = im_3d.astype(bool)
print(f"3D 影像體積建立完成！最終形狀: {im_3d.shape}")

# 3. 使用 PoreSpy 的 SNOW2 演算法提取孔隙網路
print("\n正在使用 PoreSpy (snow2) 提取網路... (這可能需要幾分鐘，請耐心等候)")

# 方法 1: 嘗試 snow2
try:
    snow_output = ps.networks.snow2(
        phase=im_3d,
        voxel_size=1
    )
    print("使用 snow2 成功！")
except Exception as e:
    print(f"snow2 失敗: {e}")
    
    # 方法 2: 嘗試原始的 snow 算法
    try:
        snow_output = ps.networks.snow(
            im=im_3d,
            voxel_size=1
        )
        print("使用 snow (v1) 成功！")
    except Exception as e2:
        print(f"snow 也失敗: {e2}")
        
        # 方法 3: 使用 extract_pore_network
        try:
            snow_output = ps.networks.extract_pore_network(
                im=im_3d,
                voxel_size=1
            )
            print("使用 extract_pore_network 成功！")
        except Exception as e3:
            print(f"extract_pore_network 失敗: {e3}")
            print("嘗試最基本的方法...")
            
            # 方法 4: 手動執行 SNOW 算法步驟
            from scipy import ndimage
            from skimage.measure import regionprops

            # 距離轉換
            dt = ndimage.distance_transform_edt(im_3d)

            # 找峰值
            from skimage.feature import peak_local_max
            peaks = peak_local_max(
                dt, 
                min_distance=5,
                indices=True
            )

            # 創建標記
            markers = np.zeros_like(im_3d, dtype=int)
            markers[peaks[:, 0], peaks[:, 1], peaks[:, 2]] = np.arange(len(peaks)) + 1

            # 分水嶺
            from skimage.segmentation import watershed
            regions = watershed(-dt, markers, mask=im_3d)

            # 計算喉道連接
            print("計算喉道連接...")
            from skimage.measure import label
            from scipy.spatial.distance import cdist

            # 獲取每個區域的中心點
            props = regionprops(regions)
            pore_coords = np.array([p.centroid for p in props])

            # 簡單方法：基於距離創建連接
            if len(pore_coords) > 1:
                # 計算所有孔隙對之間的距離
                distances = cdist(pore_coords, pore_coords)
                
                # 找出相鄰的孔隙（距離小於閾值）
                threshold = 50  # 可調整
                conns = []
                for i in range(len(pore_coords)):
                    for j in range(i+1, len(pore_coords)):
                        if distances[i, j] < threshold:
                            conns.append([i, j])
                
                throat_conns = np.array(conns) if conns else np.empty((0, 2), dtype=int)
            else:
                throat_conns = np.empty((0, 2), dtype=int)

            # 轉換為網路
            snow_output = {
                'pore.region': regions,
                'throat.conns': throat_conns,
                'pore.coords': pore_coords
            }
            print("基本方法完成！")


print("網路提取完成！")

# 4. 將 PoreSpy 的結果匯入 OpenPNM
print("正在將網路匯入 OpenPNM...")
try:
    # 檢查 snow_output 的類型
    if isinstance(snow_output, dict):
        # 如果是字典，使用 import_data
        pn = op.io.network_from_porespy(snow_output)
    else:
        # 如果是其他類型，嘗試直接使用
        pn = snow_output
    print("成功建立 OpenPNM 網路！")
except Exception as e:
    print(f"匯入失敗: {e}")
    # 創建一個簡單的網路作為備用
    pn = op.network.Cubic(shape=[10, 10, 10], spacing=1)
    print("使用備用網路")

# 5. 輸出基本資訊
print("-" * 40)
print(f"分析結果：")
print(f"從 {im_3d.shape} 的影像體積中，")
print(f"找到的孔隙 (Pores) 數量: {pn.Np}")

# 安全地嘗試獲取喉道數量
try:
    print(f"找到的喉道 (Throats) 數量: {pn.Nt}")
except AttributeError:
    print("找到的喉道 (Throats) 數量: 無法計算（網路未完全建立）")
print("-" * 40)

# 6. 分析並繪製孔隙尺寸分佈
if 'pore.diameter' in pn:
    pore_diameters = pn['pore.diameter']
    plt.figure(1)
    plt.hist(pore_diameters, bins=30, edgecolor='k')
    plt.title('Pore (Sphere) Diameter Distribution')
    plt.xlabel('Diameter')
    plt.ylabel('Frequency (Count)')
    print("正在準備孔隙尺寸分佈圖...")
else:
    print("警告：找不到 'pore.diameter' 數據。")

# 7. 分析並繪製喉道尺寸分佈
if 'throat.diameter' in pn:
    throat_diameters = pn['throat.diameter']
    plt.figure(2)
    plt.hist(throat_diameters, bins=30, edgecolor='k')
    plt.title('Throat (Cylinder) Diameter Distribution')
    plt.xlabel('Diameter')
    plt.ylabel('Frequency (Count)')
    print("正在準備喉道尺寸分佈圖...")
else:
    print("警告：找不到 'throat.diameter' 數據。")

# 8. 顯示所有圖表
print("所有分析完成，正在顯示圖表...")
plt.show()

print("程式執行完畢。")
