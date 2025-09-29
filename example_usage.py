#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenPNM 孔隙網路分析 - 使用範例

此腳本展示如何使用改善後的 test2_0921_GPU.py 中的功能
進行孔隙網路分析。
"""

import porespy as ps
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_sample_network():
    """
    創建一個簡單的示例網路用於演示
    """
    print("創建示例網路...")
    
    # 創建簡單的立方網路
    pn = op.network.Cubic(shape=[5, 5, 5], spacing=10e-6)  # 10微米間距
    
    # 設定孔隙直徑
    pn['pore.diameter'] = np.random.normal(10e-6, 2e-6, pn.Np)  # 10±2微米
    pn['pore.diameter'][pn['pore.diameter'] < 5e-6] = 5e-6  # 最小5微米
    
    # 設定喉道直徑
    pn['throat.diameter'] = np.random.normal(5e-6, 1e-6, pn.Nt)  # 5±1微米
    pn['throat.diameter'][pn['throat.diameter'] < 2e-6] = 2e-6  # 最小2微米
    
    # 計算孔隙體積（球體）
    pn['pore.volume'] = (4/3) * np.pi * (pn['pore.diameter']/2)**3
    
    # 計算喉道面積（圓形）
    pn['throat.area'] = np.pi * (pn['throat.diameter']/2)**2
    
    print(f"網路創建完成：{pn.Np} 孔隙，{pn.Nt} 喉道")
    return pn

def analyze_network_properties(network):
    """
    分析網路屬性
    """
    print("\n=== 網路屬性分析 ===")
    
    # 基本資訊
    print(f"孔隙數量: {network.Np}")
    print(f"喉道數量: {network.Nt}")
    print(f"平均協調數: {network.Nt * 2 / network.Np:.2f}")
    
    # 幾何屬性
    if 'pore.diameter' in network:
        pore_d = network['pore.diameter'] * 1e6  # 轉換為微米
        print(f"孔隙直徑: {pore_d.mean():.2f} ± {pore_d.std():.2f} μm")
    
    if 'throat.diameter' in network:
        throat_d = network['throat.diameter'] * 1e6  # 轉換為微米
        print(f"喉道直徑: {throat_d.mean():.2f} ± {throat_d.std():.2f} μm")
    
    return {
        'n_pores': network.Np,
        'n_throats': network.Nt,
        'coordination': network.Nt * 2 / network.Np
    }

def run_flow_simulation(network):
    """
    運行流動模擬
    """
    print("\n=== 流動模擬 ===")
    
    try:
        # 創建水的相位
        water = op.phase.Water(network=network)
        
        # 創建 Stokes 流動演算法
        sf = op.algorithms.StokesFlow(network=network, phase=water)
        
        # 設定邊界條件
        inlet = network.pores('left')
        outlet = network.pores('right')
        
        if len(inlet) > 0 and len(outlet) > 0:
            sf.set_value_BC(pores=inlet, values=101325)  # 入口壓力
            sf.set_value_BC(pores=outlet, values=101325-1000)  # 出口壓力差1000Pa
            
            # 運行模擬
            sf.run()
            
            # 計算流率
            flow_rate = abs(sf.rate(pores=inlet)[0])
            print(f"流率: {flow_rate*1e9:.2f} nL/s")
            
            return flow_rate
        else:
            print("無法設定邊界條件（沒有左右邊界）")
            return None
            
    except Exception as e:
        print(f"流動模擬失敗: {e}")
        return None

def visualize_network(network):
    """
    可視化網路
    """
    print("\n=== 網路可視化 ===")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 子圖1：孔隙直徑分佈
    ax1 = plt.subplot(1, 3, 1)
    if 'pore.diameter' in network:
        pore_d = network['pore.diameter'] * 1e6
        plt.hist(pore_d, bins=20, edgecolor='k', alpha=0.7)
        plt.xlabel('孔隙直徑 (μm)')
        plt.ylabel('數量')
        plt.title('孔隙直徑分佈')
        plt.grid(True, alpha=0.3)
    
    # 子圖2：喉道直徑分佈
    ax2 = plt.subplot(1, 3, 2)
    if 'throat.diameter' in network:
        throat_d = network['throat.diameter'] * 1e6
        plt.hist(throat_d, bins=20, edgecolor='k', alpha=0.7, color='orange')
        plt.xlabel('喉道直徑 (μm)')
        plt.ylabel('數量')
        plt.title('喉道直徑分佈')
        plt.grid(True, alpha=0.3)
    
    # 子圖3：3D 網路結構
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    coords = network['pore.coords']
    
    # 繪製孔隙
    if 'pore.diameter' in network:
        sizes = network['pore.diameter'] * 1e6 * 100  # 放大顯示
        scatter = ax3.scatter(coords[:, 0]*1e6, coords[:, 1]*1e6, coords[:, 2]*1e6,
                             s=sizes, alpha=0.6, c=network['pore.diameter']*1e6, cmap='viridis')
        plt.colorbar(scatter, ax=ax3, shrink=0.5, label='孔隙直徑 (μm)')
    
    # 繪製部分喉道連接
    conns = network['throat.conns'][:50]  # 只顯示前50個連接
    for conn in conns:
        x_vals = [coords[conn[0], 0]*1e6, coords[conn[1], 0]*1e6]
        y_vals = [coords[conn[0], 1]*1e6, coords[conn[1], 1]*1e6]
        z_vals = [coords[conn[0], 2]*1e6, coords[conn[1], 2]*1e6]
        ax3.plot(x_vals, y_vals, z_vals, 'k-', alpha=0.3, linewidth=0.5)
    
    ax3.set_xlabel('X (μm)')
    ax3.set_ylabel('Y (μm)')
    ax3.set_zlabel('Z (μm)')
    ax3.set_title('3D 網路結構')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函數
    """
    print("OpenPNM 孔隙網路分析 - 使用範例")
    print("="*50)
    
    # 1. 創建示例網路
    network = create_sample_network()
    
    # 2. 分析網路屬性
    props = analyze_network_properties(network)
    
    # 3. 運行流動模擬
    flow_rate = run_flow_simulation(network)
    
    # 4. 可視化結果
    visualize_network(network)
    
    # 5. 匯出資訊
    print("\n=== 分析總結 ===")
    print(f"這是一個包含 {props['n_pores']} 個孔隙和 {props['n_throats']} 個喉道的網路")
    print(f"平均協調數為 {props['coordination']:.2f}")
    if flow_rate:
        print(f"在1000Pa壓差下的流率為 {flow_rate*1e9:.2f} nL/s")
    
    print("\n✓ 分析完成！")
    print("這個範例展示了如何使用 OpenPNM 進行基本的孔隙網路分析。")
    print("更多功能請參考 test2_0921_GPU.py 中的完整實現。")

if __name__ == "__main__":
    main()