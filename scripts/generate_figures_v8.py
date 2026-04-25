#!/usr/bin/env python3
"""
논문 v8용 핵심 그림 재생성.

출력:
  figures/v8_fig4_shap_bar.png  — Top 15 변수 중요도
  figures/v8_fig5_shap_summary.png — 전체 SHAP Summary (beeswarm)
  figures/v8_fig6_dep_건물연령.png — U자형 Dependence
  figures/v8_fig7_dep_전용면적.png — 체감형 Dependence
  figures/v8_fig8_dep_subway_nearest.png — 임계반응형 Dependence
  figures/v8_fig9_dep_department_nearest.png — 백화점 최근접 Dependence
  figures/v8_fig10_ablation.png — 시나리오별 R² 비교 바차트
  figures/v8_fig11_region_shap.png — 강남 vs 비강남 Top5 비교
  figures/v8_fig12_year_region_heatmap.png — 연도×권역 R² 히트맵
  figures/v8_fig13_top1_timeline.png — 연도별 Top1 변화 라인
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import shap
import sys
sys.path.insert(0, os.path.dirname(__file__))
from generate_figures import LABEL_MAP, kr

# 스타일과 동일: NanumGothic(공백명), 300dpi, 학술 팔레트
_korean_fonts = [f.name for f in font_manager.fontManager.ttflist]
for _cand in ['NanumGothic', 'Nanum Gothic', 'AppleGothic', 'Apple SD Gothic Neo']:
    if _cand in _korean_fonts:
        plt.rcParams['font.family'] = _cand
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# v7 색상 팔레트
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_ACCENT = '#2ca02c'
COLOR_GANGNAM = '#d62728'
COLOR_NON_GANGNAM = '#1f77b4'
COLOR_LIGHT = '#e8e8e8'

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(FIGS, exist_ok=True)

OLS_FEATS = [
    '전용면적', '층', '건물연령', '강남구분',
    'subway_nearest_m', 'elem_school_nearest_m', 'middle_school_nearest_m',
    'library_nearest_m', 'park_nearest_m', 'mart_nearest_m',
    'department_nearest_m', 'academy_nearest_m', 'hospital_general_nearest_m',
    'childcare_count_1000m', 'cctv_count_500m',
    'park_within_1km', 'department_within_1km',
    '기준금리', '소비자물가지수', 'M2',
]
TREE_FEATS = OLS_FEATS + [
    'subway_count_1000m', 'elem_school_count_1000m', 'middle_school_count_1000m',
    'high_school_count_1000m', 'mart_count_1000m', 'department_count_2000m',
    'academy_count_1000m', 'library_count_1000m', 'hospital_count_1000m',
    'park_count_1000m', 'large_store_count_500m',
    'park_log1p_count_2km', 'department_log1p_count_2km', 'library_log1p_count_2km',
    'CD금리',
]


def load_v8():
    df = pd.read_csv(os.path.join(ROOT, 'data/apartment_final_v8.csv'), low_memory=False)
    df = df.dropna(subset=TREE_FEATS + ['log㎡당가격'])
    return df


def fit_xgb(df, feats):
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    m = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     n_jobs=-1, random_state=42, tree_method='hist')
    m.fit(tr[feats], tr['log㎡당가격'].values)
    return m, tr, te


def shap_values(model, sample):
    exp = shap.TreeExplainer(model)
    return exp.shap_values(sample)


def fig_shap_bar(sv, feats, out):
    abs_mean = np.abs(sv).mean(axis=0)
    order = np.argsort(abs_mean)[::-1][:15]
    names = [feats[i] for i in order]
    vals = abs_mean[order]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(names)), vals[::-1], color='#1f77b4')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([kr(n) for n in names[::-1]])
    ax.set_xlabel('평균 |SHAP| (log 스케일)')
    ax.set_title('<그림 4-1> SHAP Bar — 전체 모형 Top 15')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_shap_summary(sv, X, feats, out):
    # shap의 summary_plot을 저장. feature_names를 한글로 매핑
    plt.figure(figsize=(9, 7))
    Xkr = X[feats].copy()
    Xkr.columns = [kr(c) for c in Xkr.columns]
    shap.summary_plot(sv, Xkr, show=False, max_display=15)
    plt.title('<그림 4-2> SHAP Summary — 변수 방향성 및 분포')
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def fig_dependence(sv, X, feats, var_name, out, title):
    idx = feats.index(var_name)
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(idx, sv, X[feats], show=False, feature_names=feats)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def fig_ablation(out):
    rows = pd.read_csv(os.path.join(ROOT, 'results/ablation_v7_v8.csv'))
    pv = rows.pivot(index='scenario', columns='split', values='xgb_r2')
    pv = pv.reindex(['A_행정동만_v7', 'B_거리만_시점무관(2026스냅샷)', 'C_거리+시점정합()', 'D_거리+시점+행정동'])
    pv.index = ['A 행정동만(v7)', 'B 거리만·시점무관', 'C 거리+시점정합', 'D 통합(거리+시점+행정동)']
    pv = pv[['random', 'temporal', 'group']]
    pv.columns = ['무작위', '시간순', 'Group']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pv.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('XGBoost R²')
    ax.set_title('<그림 4-7> 어블레이션 시나리오별 XGB R² (분할별)')
    ax.legend(title='분할')
    ax.grid(True, axis='y', alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=8, padding=2)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_region_shap(out):
    # 권역별 SHAP Top 5 — v8_year_region_shap_top.csv 전체 기간 집계 대신
    # 각 권역의 연도별 mean을 변수별 집계
    d = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_shap_top.csv'))
    agg = d.groupby(['region', 'feature'])['mean_abs_shap'].sum().reset_index()
    top5 = {}
    for r in ['강남3구', '비강남']:
        sub = agg[agg['region'] == r].sort_values('mean_abs_shap', ascending=False).head(5)
        top5[r] = sub.set_index('feature')['mean_abs_shap']
    all_feats = list(set(top5['강남3구'].index) | set(top5['비강남'].index))
    gn = [top5['강남3구'].get(f, 0) for f in all_feats]
    bn = [top5['비강남'].get(f, 0) for f in all_feats]
    order = sorted(range(len(all_feats)), key=lambda i: max(gn[i], bn[i]), reverse=True)
    all_feats = [all_feats[i] for i in order]
    gn = [gn[i] for i in order]
    bn = [bn[i] for i in order]
    x = np.arange(len(all_feats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, gn, w, label='강남3구', color='#d62728')
    ax.bar(x + w/2, bn, w, label='비강남', color='#1f77b4')
    ax.set_xticks(x)
    ax.set_xticklabels([kr(f) for f in all_feats], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('연도 합산 평균 |SHAP|')
    ax.set_title('<그림 4-8> 권역별 SHAP 주요 변수 비교')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_year_region_heatmap(out):
    s = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_summary.csv'))
    pv = s.pivot(index='year', columns='region', values='r2')
    pv = pv[['강남3구', '비강남', '전체']]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pv.values, cmap='RdYlGn', vmin=0.85, vmax=0.97, aspect='auto')
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_title('<그림 4-9> 연도×권역 XGB R² 히트맵')
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='white' if v < 0.9 else 'black', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_top1_timeline(out):
    d = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_shap_top.csv'))
    top1 = d[d['rank'] == 1].copy()
    pv = top1.pivot(index='year', columns='region', values='feature')[['강남3구', '비강남']]
    pv.to_csv(os.path.join(ROOT, 'results/v8_top1_by_year_region.csv'))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    years = pv.index.astype(int).tolist()
    ax.plot(years, [1] * len(years), 'o', color='#d62728', markersize=14)
    ax.plot(years, [0] * len(years), 'o', color='#1f77b4', markersize=14)
    for y, f in zip(years, pv['강남3구']):
        ax.annotate(kr(f), (y, 1), textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=8, color='#d62728')
    for y, f in zip(years, pv['비강남']):
        ax.annotate(kr(f), (y, 0), textcoords="offset points", xytext=(0, -22),
                    ha='center', fontsize=8, color='#1f77b4')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['비강남', '강남3구'])
    ax.set_xticks(years)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title('<그림 4-10> 연도별 Top 1 SHAP 변수 변화 (강남3구 vs 비강남)')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    print(" 데이터 로딩...")
    df = load_v8()
    print(f"데이터: {len(df):,}")
    print("XGBoost 적합...")
    model, tr, te = fit_xgb(df, TREE_FEATS)

    sample = te.sample(min(5000, len(te)), random_state=42)
    print("SHAP 계산...")
    sv = shap_values(model, sample[TREE_FEATS])

    print("그림 생성...")
    fig_shap_bar(sv, TREE_FEATS, os.path.join(FIGS, 'v8_fig4_shap_bar.png'))
    fig_shap_summary(sv, sample, TREE_FEATS, os.path.join(FIGS, 'v8_fig5_shap_summary.png'))
    fig_dependence(sv, sample, TREE_FEATS, '건물연령',
                   os.path.join(FIGS, 'v8_fig6_dep_건물연령.png'),
                   '<그림 4-3> SHAP Dependence — 건물연령 U자형 비선형 패턴')
    fig_dependence(sv, sample, TREE_FEATS, '전용면적',
                   os.path.join(FIGS, 'v8_fig7_dep_전용면적.png'),
                   '<그림 4-4> SHAP Dependence — 전용면적 체감형 비선형 패턴')
    fig_dependence(sv, sample, TREE_FEATS, 'subway_nearest_m',
                   os.path.join(FIGS, 'v8_fig8_dep_subway_nearest.png'),
                   '<그림 4-5> SHAP Dependence — 지하철 최근접거리 임계반응형')
    fig_dependence(sv, sample, TREE_FEATS, 'department_nearest_m',
                   os.path.join(FIGS, 'v8_fig9_dep_department_nearest.png'),
                   '<그림 4-6> SHAP Dependence — 백화점 최근접거리')

    fig_ablation(os.path.join(FIGS, 'v8_fig10_ablation.png'))
    fig_region_shap(os.path.join(FIGS, 'v8_fig11_region_shap.png'))
    fig_year_region_heatmap(os.path.join(FIGS, 'v8_fig12_year_region_heatmap.png'))
    fig_top1_timeline(os.path.join(FIGS, 'v8_fig13_top1_timeline.png'))
    print(f"저장 완료: figures/v8_fig4~13")


if __name__ == '__main__':
    main()
