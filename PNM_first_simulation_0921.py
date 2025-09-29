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

try:
    # 嘗試使用 image 參數
    snow_output = ps.networks.snow2(image=im_3d, voxel_size=1)
except TypeError:
    try:
        # 如果失敗，嘗試位置參數
        snow_output = ps.networks.snow2(im_3d, 1)
    except:
        # 如果還是失敗，使用替代方法
        print("snow2 失敗，使用替代方法...")
        dt = ps.filters.distance_transform(im_3d)
        peaks = ps.filters.find_peaks(dt)
        regions = ps.filters.watershed(dt, markers=peaks)
        snow_output = ps.networks.regions_to_network(regions, voxel_size=1)

print("網路提取完成！")

# 4. 將 PoreSpy 的結果匯入 OpenPNM
print("正在將網路匯入 OpenPNM...")
try:
    # 對於 PoreSpy 2.x 和 OpenPNM 3.x
    pn = op.io.network_from_porespy(snow_output)
    print("成功建立 OpenPNM 網路！")
except:
    # 如果上面的方法失敗，嘗試其他方法
    project = op.io.PoreSpy.import_data(snow_output)
    pn = project.network
    print("成功建立 OpenPNM 專案！")

# 5. 輸出基本資訊
print("-" * 40)
print(f"分析結果：")
print(f"從 {im_3d.shape} 的影像體積中，")
print(f"找到的孔隙 (Pores) 數量: {pn.Np}")
print(f"找到的喉道 (Throats) 數量: {pn.Nt}")
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
