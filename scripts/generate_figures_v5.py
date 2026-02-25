#!/usr/bin/env python3
"""v5 논문 그림 전체 생성 - SciencePlots 스타일, 영어 텍스트"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex', 'grid'])
import matplotlib.font_manager as fm

# 영어 레이블 매핑
FEATURE_EN = {
    '전용면적': 'Exclusive Area',
    '층': 'Floor',
    '건물연령': 'Building Age',
    '강남구분': 'Gangnam Dummy',
    '초등학교수': 'Elementary Schools',
    '중학교수': 'Middle Schools',
    '고등학교수': 'High Schools',
    'CCTV수': 'CCTV Count',
    '백화점수': 'Department Stores',
    '지하철역수': 'Subway Stations',
    '기준금리': 'Base Rate',
    'CD금리': 'CD Rate',
    '소비자물가지수': 'CPI',
    'M2': 'M2 Money Supply',
    '거래금액': 'Transaction Price',
}

FEATURES = ['전용면적', '층', '건물연령', '강남구분', '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수', '기준금리', 'CD금리', '소비자물가지수', 'M2']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def load_results():
    with open(os.path.join(RESULTS_DIR, 'modeling_v5_dong_results.json')) as f:
        return json.load(f)

def fig4_shap_bar(results):
    """SHAP Feature Importance Bar Chart"""
    shap_data = results['SHAP']
    features = sorted(shap_data.keys(), key=lambda x: shap_data[x]['pct'], reverse=True)
    pcts = [shap_data[f]['pct'] for f in features]
    labels = [FEATURE_EN.get(f, f) for f in features]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(range(len(features)), pcts, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('SHAP Contribution (%)', fontsize=12)
    ax.set_title('Feature Importance by Mean |SHAP Value|', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_shap_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig4_shap_importance.png")

def fig5_model_comparison(results):
    """Model Performance Comparison"""
    models = ['OLS', 'RF', 'XGBoost']
    r2_vals = [results[m]['R2'] for m in models]
    rmse_vals = [results[m]['RMSE'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#5C6BC0', '#66BB6A', '#EF5350']
    
    bars1 = ax1.bar(models, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('(a) R² Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars1, r2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    bars2 = ax2.bar(models, [r/1000 for r in rmse_vals], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('RMSE (×1,000 만원)', fontsize=12)
    ax2.set_title('(b) RMSE Comparison', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:,.0f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig5_model_comparison.png")

def fig6_gangnam_comparison(results):
    """Gangnam vs Non-Gangnam SHAP Comparison"""
    comp = results.get('gangnam_comparison', {})
    if not comp:
        print("  No gangnam comparison data")
        return
    
    # Sort by gangnam SHAP
    features = sorted(comp.keys(), key=lambda x: comp[x]['gangnam'], reverse=True)[:10]
    gangnam_vals = [comp[f]['gangnam']/1000 for f in features]
    non_gangnam_vals = [comp[f]['non_gangnam']/1000 for f in features]
    labels = [FEATURE_EN.get(f, f) for f in features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gangnam_vals, width, label='Gangnam 3 Districts',
                   color='#EF5350', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, non_gangnam_vals, width, label='Non-Gangnam',
                   color='#5C6BC0', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Mean |SHAP Value| (×1,000)', fontsize=12)
    ax.set_title('SHAP Value Comparison: Gangnam vs Non-Gangnam', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_gangnam_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig6_gangnam_comparison.png")

def fig7_shap_plots():
    """SHAP Summary & Dependence Plots (requires model re-run or saved SHAP values)"""
    # Use existing plots from plots_v5_dong if available
    import shutil
    src_dir = os.path.join(RESULTS_DIR, 'plots_v5_dong')
    if os.path.exists(src_dir):
        mappings = {
            'fig4_shap_summary.png': 'fig7_shap_summary.png',
            'fig5_shap_bar.png': 'fig8_shap_bar_detail.png',
        }
        for src, dst in mappings.items():
            src_path = os.path.join(src_dir, src)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(FIG_DIR, dst))
                print(f"  {dst} (copied from plots_v5_dong)")
        
        # Copy dependence plots
        dep_files = [f for f in os.listdir(src_dir) if f.startswith('fig') and 'dep_' in f]
        for i, f in enumerate(sorted(dep_files)):
            dst = f'fig{9+i}_dep_{f.split("dep_")[1]}'
            shutil.copy2(os.path.join(src_dir, f), os.path.join(FIG_DIR, dst))
            print(f"  {dst}")
        
        # Force plot
        force = os.path.join(src_dir, 'fig12_force_plot.png')
        if os.path.exists(force):
            shutil.copy2(force, os.path.join(FIG_DIR, 'fig15_force_plot.png'))
            print("  fig15_force_plot.png")

def fig_yearly_trend():
    """Yearly Transaction Trend"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v5_dong.csv'))
    
    yearly = df.groupby('거래년도').agg(
        count=('거래금액', 'size'),
        mean_price=('거래금액', 'mean')
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = '#1565C0'
    ax1.bar(yearly['거래년도'], yearly['count']/1000, color=color1, alpha=0.6, label='Transaction Count')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Transactions (×1,000)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#C62828'
    ax2.plot(yearly['거래년도'], yearly['mean_price']/10000, color=color2, marker='o',
             linewidth=2, markersize=6, label='Mean Price')
    ax2.set_ylabel('Mean Price (억원)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('Seoul Apartment Transactions by Year (2019-2025)', fontsize=13, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig16_yearly_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig16_yearly_trend.png")

def main():
    results = load_results()
    
    print("=== Generating Publication Figures ===")
    fig4_shap_bar(results)
    fig5_model_comparison(results)
    fig6_gangnam_comparison(results)
    fig7_shap_plots()
    fig_yearly_trend()
    
    # List all figures
    print(f"\n=== All figures in {FIG_DIR} ===")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith(('.png', '.svg')):
            size = os.path.getsize(os.path.join(FIG_DIR, f))
            print(f"  {f} ({size//1024}KB)")

if __name__ == '__main__':
    main()
