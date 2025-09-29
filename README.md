# 孔隙網路分析系統 (Pore Network Analysis System)

## 專案簡介

這是一個基於 Python 的3D孔隙網路分析系統，專門用於分析多孔材料的微觀結構。系統使用先進的影像處理技術和網路分析算法，能夠從3D影像資料中提取和分析孔隙網路的幾何特徵。

## 主要功能

### 🔬 影像處理與分析
- **3D影像重建**：從2D切片影像堆疊重建3D結構
- **SNOW演算法**：SubNetwork of an Oversegmented Watershed 孔隙網路提取
- **GPU加速**：支援CUDA加速的距離轉換計算
- **高精度分割**：使用分水嶺算法進行孔隙分割

### 📊 網路特徵分析
- **孔隙特徵**：體積、直徑、形狀係數
- **喉道特徵**：長度、直徑、連通性
- **網路拓撲**：連接度、配位數分析
- **孔隙率計算**：整體和局部孔隙率統計

### 🎯 改進的計算方法
- **亞像素級插值**：提高喉道直徑計算精度
- **連續值計算**：減少離散化效應
- **高密度取樣**：提升幾何特徵解析度

### 📈 可視化與報告
- **3D網路可視化**：孔隙球體和連接線3D展示
- **統計分析圖表**：直徑分佈、連接性分析
- **互動式圖表**：詳細的分析結果展示
- **連接性熱圖**：網路拓撲結構視覺化

## 技術架構

### 核心依賴
- **PoreSpy**：孔隙網路分析
- **OpenPNM**：孔隙網路建模
- **SciPy/NumPy**：科學計算
- **Scikit-image**：影像處理
- **Matplotlib**：資料視覺化
- **CuPy**：GPU加速計算（可選）

### 系統需求
- Python 3.8+
- 8GB+ RAM（建議16GB以上）
- NVIDIA GPU（可選，用於加速）
- Windows/Linux/macOS

## 安装與使用

### 1. 克隆專案
```bash
git clone https://github.com/your-username/pore-network-analysis.git
cd pore-network-analysis
```

### 2. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依賴
```bash
pip install -r requirements.txt
```

### 4. GPU支援（可選）
```bash
pip install cupy-cuda11x  # 根據CUDA版本選擇
```

## 使用範例

### 基本分析流程
```python
import porespy as ps
import openpnm as op
import numpy as np

# 設定參數
PIXEL_SIZE = 0.65  # 微米/像素

# 讀取影像資料
# 程式會自動搜尋 smallpore_*_2Dtiff/ 目錄中的影像

# 執行分析
python test2_0921_GPU.py
```

### 主要輸出
1. **網路統計資訊**：孔隙數量、喉道數量、平均連接度
2. **幾何分析**：直徑分佈、長度統計
3. **3D可視化**：網路結構、連接性分析
4. **改進效果**：離散度提升、精度改善

## 檔案結構

```
cmp_analysis/
├── test2_0921_GPU.py           # 主要分析程式（GPU加速版本）
├── test2_0921.py               # CPU版本分析程式
├── Particle_analysis_*.py      # 粒子分析相關程式
├── Example for image segmentation.py  # 影像分割範例
├── smallpore_*_2Dtiff/         # 影像資料目錄
├── requirements.txt            # Python依賴清單
├── README.md                  # 專案說明文件
└── .gitignore                # Git忽略檔案配置
```

## 演算法特點

### SNOW演算法改進
1. **距離轉換優化**：使用GPU加速的歐氏距離轉換
2. **峰值檢測增強**：自適應最小距離參數
3. **分水嶺分割**：低緊致性參數以保持自然形狀
4. **網路提取**：基於KDTree的高效連接分析

### 喉道直徑計算改進
- **問題**：原始方法產生離散化效應
- **解決方案**：
  - RegularGridInterpolator亞像素插值
  - 3倍密度取樣
  - 微量隨機擾動避免重複值
  - 連續值計算提升精度

## 性能優化

### GPU加速特性
- **自動檢測**：程式會自動檢測GPU可用性
- **記憶體管理**：智能分塊處理大型資料集
- **降級機制**：GPU不可用時自動切換到CPU

### 記憶體優化
- **批次處理**：分批讀取大型影像集
- **垃圾回收**：及時釋放不需要的記憶體
- **資料類型優化**：使用適當的數值精度

## 輸出結果

### 統計報告
- 孔隙直徑分佈統計（最小值、最大值、平均值、中位數、標準差）
- 喉道特徵統計（長度範圍、直徑範圍、長徑比）
- 網路連接性分析（連接數分佈、配位數）
- 改進效果量化（唯一值比例、解析度指標）

### 可視化輸出
- 孔隙尺寸分佈直方圖
- 喉道幾何特徵散點圖  
- 3D網路結構圖
- 連接性分析圖

## 版本歷史

### v1.2.0 (最新)
- ✅ 添加亞像素級喉道直徑計算
- ✅ 新增3D網路可視化功能
- ✅ 改善統計分析和報告
- ✅ GPU加速優化

### v1.1.0
- ✅ 實現SNOW演算法
- ✅ GPU加速距離轉換
- ✅ 基本網路分析功能

### v1.0.0
- ✅ 基礎影像處理功能
- ✅ 孔隙網路提取

## 貢獻指南

歡迎提交Issue和Pull Request！

### 開發流程
1. Fork專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟Pull Request

## 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 聯絡方式

- **作者**：bohan
- **郵箱**：s6b0753@gmail.com
- **專案連結**：[https://github.com/your-username/pore-network-analysis](https://github.com/your-username/pore-network-analysis)

## 致謝

感謝以下開源專案的支持：
- [PoreSpy](https://github.com/PMEAL/porespy)
- [OpenPNM](https://github.com/PMEAL/OpenPNM)
- [SciPy](https://scipy.org/)
- [Scikit-image](https://scikit-image.org/)

---

**注意**：本專案仍在積極開發中，歡迎反饋和建議！