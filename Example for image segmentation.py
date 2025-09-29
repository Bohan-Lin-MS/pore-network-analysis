import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_medical_image_example():
    """創建模擬的醫學影像和對應的標註"""
    # 創建 512x512 的基礎影像
    img_size = 512
    image = np.zeros((img_size, img_size), dtype=np.float32)
    
    # 添加背景紋理（模擬組織結構）
    np.random.seed(42)
    background = np.random.normal(0.3, 0.1, (img_size, img_size))
    background = gaussian_filter(background, sigma=2)
    image += np.clip(background, 0, 1)
    
    # 創建目標結構（模擬血管或器官）
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    
    # 主要結構 1：彎曲的管狀結構
    y, x = np.ogrid[:img_size, :img_size]
    for t in np.linspace(0, 4*np.pi, 200):
        center_x = int(256 + 80 * np.sin(t))
        center_y = int(100 + t * 30)
        if 0 <= center_x < img_size and 0 <= center_y < img_size:
            radius = 15 + 5 * np.sin(2*t)
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask[dist <= radius] = 1
    
    # 主要結構 2：不規則塊狀結構
    center_x, center_y = 350, 300
    for angle in np.linspace(0, 2*np.pi, 100):
        radius = 40 + 20 * np.sin(3*angle) + 10 * np.cos(5*angle)
        x_pos = int(center_x + radius * np.cos(angle))
        y_pos = int(center_y + radius * np.sin(angle))
        if 0 <= x_pos < img_size and 0 <= y_pos < img_size:
            cv2.circle(mask, (x_pos, y_pos), 8, 1, -1)
    
    # 添加小的分支結構
    for i in range(5):
        start_x = np.random.randint(50, img_size-50)
        start_y = np.random.randint(50, img_size-50)
        end_x = start_x + np.random.randint(-100, 100)
        end_y = start_y + np.random.randint(-100, 100)
        cv2.line(mask, (start_x, start_y), (end_x, end_y), 1, thickness=8)
    
    # 平滑處理
    mask = gaussian_filter(mask.astype(float), sigma=1.5)
    mask = (mask > 0.3).astype(float)
    
    # 在影像中添加目標結構（較亮的區域）
    structure_intensity = 0.7 + 0.2 * np.random.random(mask.shape)
    image[mask > 0] = structure_intensity[mask > 0]
    
    # 添加雜訊
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # 模糊邊界（模擬實際醫學影像的挑戰）
    image = gaussian_filter(image, sigma=0.8)
    
    return image, mask

# 創建範例資料
original_image, ground_truth = create_medical_image_example()

print("✅ 已創建模擬醫學影像")
print(f"影像尺寸: {original_image.shape}")
print(f"像素值範圍: {original_image.min():.3f} - {original_image.max():.3f}")
print(f"目標區域佔比: {(ground_truth > 0).sum() / ground_truth.size * 100:.1f}%")
