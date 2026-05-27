#!/usr/bin/env python3
"""
논문용 핵심 그림 생성 (학술 스타일·한글 라벨).

출력:
  figures/fig4_shap_bar.png  — 상위 15개 변수 중요도
  figures/fig5_shap_summary.png — 전체 SHAP 요약도 (beeswarm)
  figures/fig6_dep_건물연령.png — U자형 Dependence
  figures/fig7_dep_childcare_count_1000m.png — 생활SOC 밀도 Dependence
  figures/fig8_dep_subway_nearest.png — 임계반응형 Dependence
  figures/fig9_dep_department_nearest.png — 백화점 최근접 Dependence
  figures/fig10_ablation.png — 시나리오별 R² 비교 바차트
  figures/fig11_region_shap.png — 강남 vs 비강남 상위 5개 비교
  figures/fig12_year_region_heatmap.png — 연도×권역 R² 히트맵
  figures/fig13_top1_timeline.png — 연도별 최상위 변수 변화 라인
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib import font_manager
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import shap

# 학술 논문용 시각 스타일 (SciencePlots: science + grid + no-latex)
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'grid', 'no-latex'])
except Exception:
    plt.style.use('seaborn-v0_8-paper')
# 학술 스타일이 unicode_minus를 다시 켜는 경우가 있어 명시 해제
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 변수명 한글 라벨 매핑 (모든 그림에서 일관 적용)
LABEL_MAP = {
    # 거리 기반 — 지하철
    'subway_nearest_m': '지하철 최근접(m)',
    'subway_count_500m': '지하철 500m 내', 'subway_count_1000m': '지하철 1km 내',
    'subway_count_2000m': '지하철 2km 내',
    # 학교
    'elem_school_nearest_m': '초등학교 최근접(m)',
    'elem_school_count_500m': '초등 500m 내', 'elem_school_count_1000m': '초등 1km 내',
    'middle_school_nearest_m': '중학교 최근접(m)',
    'middle_school_count_1000m': '중학교 1km 내',
    'high_school_nearest_m': '고등학교 최근접(m)',
    'high_school_count_1000m': '고등학교 1km 내',
    # 어린이집·학원
    'childcare_count_500m': '어린이집 500m 내',
    'childcare_count_1000m': '어린이집 1km 내',
    'academy_nearest_m': '학원 최근접(m)', 'academy_count_1000m': '학원 1km 내',
    # 공원·도서관
    'park_nearest_m': '공원 최근접(m)', 'park_count_1000m': '공원 1km 내',
    'park_within_1km': '공원 1km 더미', 'park_log1p_count_2km': '공원 log(2km개수)',
    'library_nearest_m': '도서관 최근접(m)', 'library_count_1000m': '도서관 1km 내',
    'library_log1p_count_2km': '도서관 log(2km개수)',
    # 상업·의료
    'mart_nearest_m': '대형마트 최근접(m)', 'mart_count_1000m': '대형마트 1km 내',
    'department_nearest_m': '백화점 최근접(m)',
    'department_within_1km': '백화점 1km 더미',
    'department_count_2000m': '백화점 2km 내',
    'department_log1p_count_2km': '백화점 log(2km개수)',
    'hospital_nearest_m': '병원 최근접(m)',
    'hospital_count_1000m': '병원 1km 내',
    'hospital_general_nearest_m': '종합병원 최근접(m)',
    # 안전·근린
    'cctv_count_500m': 'CCTV 500m 내',
    'large_store_count_500m': '근린시설 500m 내',
    # 행정동 집계 기준선 컬럼명
    '초등학교수_admin_count': '초등학교수(행정동)', '중학교수_admin_count': '중학교수(행정동)',
    '고등학교수_admin_count': '고등학교수(행정동)', 'CCTV수_admin_count': 'CCTV수(행정동)',
    '백화점수_admin_count': '백화점수(행정동)', '지하철역수_admin_count': '지하철역수(행정동)',
    '공원수_admin_count': '공원수(행정동)', '도서관수_admin_count': '도서관수(행정동)',
    '학원수_admin_count': '학원수(행정동)', '어린이집수_admin_count': '어린이집수(행정동)',
    # 거시
    '기준금리': '기준금리(%)', 'CD금리': 'CD금리(%)',
    '소비자물가지수': '소비자물가지수', 'M2': 'M2 통화량',
    # 물리·권역
    '전용면적': '전용면적(㎡)', '층': '층', '건물연령': '건물연령(년)',
    '강남구분': '강남3구 여부',
}


def kr(name):
    return LABEL_MAP.get(name, name)


def feature_category(name):
    if name == '강남구분':
        return '권역'
    if name in {'층', '건물연령'}:
        return '물리'
    if name in {'기준금리', 'CD금리', '소비자물가지수', 'M2'}:
        return '거시'
    if name.endswith('_admin_count') or name.endswith('수_admin_count'):
        return '행정동'
    return '거리'

# 한글 폰트 (학술 스타일에 한글 cascade 적용)
# 본문 docx와 동일하게 KoPubWorld Batang 우선 — 가이드 명조계열 통일.
# 사용자 폰트 디렉토리(~/Library/Fonts)도 함께 스캔하여 KoPubWorld OTF 인식.
import os as _os
_extra_fonts = font_manager.findSystemFonts(
    fontpaths=[_os.path.expanduser('~/Library/Fonts')]
)
for _fp in _extra_fonts:
    if _fp not in {f.fname for f in font_manager.fontManager.ttflist}:
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
_korean_fonts = {f.name for f in font_manager.fontManager.ttflist}
_kr_font = None
for _cand in [
    'KoPubWorldBatang_Pro',  # 본문과 동일 (명조계열 — 가이드 권장)
    'AppleMyungjo',          # macOS 기본 명조 fallback
    'Nanum Myeongjo', 'NanumMyeongjo',
    'NanumGothic', 'Nanum Gothic', 'AppleGothic', 'Apple SD Gothic Neo',
]:
    if _cand in _korean_fonts:
        _kr_font = _cand
        break
if _kr_font:
    plt.rcParams['font.family'] = _kr_font
    plt.rcParams['mathtext.fontset'] = 'cm'  # 수식 폰트는 Computer Modern (학술)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10

# v7 색상 팔레트
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_ACCENT = '#2ca02c'
COLOR_GANGNAM = '#d62728'
COLOR_NON_GANGNAM = '#1f77b4'
COLOR_LIGHT = '#e8e8e8'
CATEGORY_COLORS = {
    '권역': '#7E22CE',
    '물리': '#0E7490',
    '거시': '#475569',
    '거리': '#15803D',
    '행정동': '#C2410C',
}

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(FIGS, exist_ok=True)

OLS_FEATS = [
    '층', '건물연령', '강남구분',
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


def decorate_dependence(ax, var_name):
    """논문 해석에 필요한 임계구간을 dependence plot 위에 표시."""
    guides = {
        'childcare_count_1000m': [(40, '40개')],
        'department_nearest_m': [(1000, '1km'), (3000, '3km\n기여 소멸권')],
    }
    spans = {
        '건물연령': [(25, 30, '25~30년\n재건축 기대 구간')],
        'subway_nearest_m': [(300, 500, '300~500m\n역세권 임계')],
    }
    for x0, x1, label in spans.get(var_name, []):
        ax.axvspan(x0, x1, color='#FEE2E2', alpha=0.35, zorder=0)
        ax.axvline(x0, color='#DC2626', linestyle='--', linewidth=1.0, alpha=0.75)
        ax.axvline(x1, color='#DC2626', linestyle='--', linewidth=1.0, alpha=0.75)
        ymin, ymax = ax.get_ylim()
        ax.text(
            (x0 + x1) / 2,
            ymax - (ymax - ymin) * 0.07,
            label,
            color='#991B1B',
            fontsize=8,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#FCA5A5', alpha=0.88),
        )
    for x, label in guides.get(var_name, []):
        ax.axvline(x, color='#DC2626', linestyle='--', linewidth=1.0, alpha=0.75)
        ymin, ymax = ax.get_ylim()
        ax.text(
            x,
            ymax - (ymax - ymin) * 0.07,
            label,
            color='#991B1B',
            fontsize=8,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#FCA5A5', alpha=0.88),
        )


def tune_axes(ax, tick_size=7.5, label_size=8.5, title_size=9.5):
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)


def wrap_feature_label(label):
    return (label
            .replace('(행정동)', '\n(행정동)')
            .replace('최근접(m)', '\n최근접(m)')
            .replace('1km 내', '\n1km 내')
            .replace('500m 내', '\n500m 내')
            .replace('전용면적(㎡)', '전용면적\n(㎡)')
            .replace('건물연령(년)', '건물연령\n(년)'))


def fig_shap_bar(sv, feats, out):
    abs_mean = np.abs(sv).mean(axis=0)
    order = np.argsort(abs_mean)[::-1][:15]
    names = [kr(feats[i]) for i in order]
    vals = abs_mean[order]
    cats = [feature_category(feats[i]) for i in order]
    colors = [CATEGORY_COLORS[c] for c in cats]
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=7.2)
    ax.set_xlabel('평균 |SHAP| (log 스케일)')
    ax.set_title('전체 모형 SHAP 변수 중요도: 상위 15개')
    ax.grid(True, axis='x', alpha=0.3)
    handles = [
        Patch(facecolor=CATEGORY_COLORS[c], label=c)
        for c in ['권역', '물리', '거시', '거리', '행정동']
        if c in cats
    ]
    ax.legend(handles=handles, title='변수 범주', loc='lower right', frameon=True, fontsize=6.8, title_fontsize=7.0)
    tune_axes(ax, tick_size=7.2, label_size=8.3, title_size=9.5)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_shap_summary(sv, X, feats, out):
    # shap의 summary_plot을 저장 (한글 라벨 적용)
    plt.figure(figsize=(5.5, 4.8))
    Xkr = X[feats].rename(columns={c: kr(c) for c in feats})
    shap.summary_plot(sv, Xkr, show=False, max_display=15)
    plt.title('SHAP 요약도: 변수값 방향성과 기여 분포')
    for ax in plt.gcf().axes:
        tune_axes(ax, tick_size=6.8, label_size=7.8, title_size=9.2)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def fig_dependence(sv, X, feats, var_name, out, title):
    idx = feats.index(var_name)
    plt.figure(figsize=(5.5, 3.9))
    feats_kr = [kr(f) for f in feats]
    shap.dependence_plot(idx, sv, X[feats], show=False, feature_names=feats_kr)
    ax = plt.gca()
    decorate_dependence(ax, var_name)
    plt.title(title)
    for a in plt.gcf().axes:
        tune_axes(a, tick_size=7.0, label_size=8.0, title_size=9.2)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def fig_ablation(out):
    rows = pd.read_csv(os.path.join(ROOT, 'results/ablation_v7_v8.csv'))
    pv = rows.pivot(index='scenario', columns='split', values='xgb_r2')
    pv = pv.reindex(['A_행정동집계_기준선', 'B_거리만_시점무관(2026스냅샷)', 'C_거리+시점정합_비교모형'])
    pv.index = ['A 행정동 집계 기준선', 'B 거리만·시점무관', 'C 거리+시점정합 비교모형']
    pv = pv[['random', 'temporal', 'group']]
    pv.columns = ['무작위', '시간순', '단지']
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    pv.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('XGBoost R²')
    ax.set_title('Ablation 시나리오별 XGBoost R²')
    ax.legend(title='분할')
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.02,
        0.94,
        '핵심: 거리 기반 전환은 무작위·시간순 성능을 개선\n시점 정합은 성능보다 시간역전 정보누수 제거가 목적',
        transform=ax.transAxes,
        fontsize=6.8,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.32', facecolor='white', edgecolor='#CBD5E1', alpha=0.92),
    )
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=6.6, padding=2)
    ax.legend(title='분할', fontsize=7, title_fontsize=7)
    tune_axes(ax, tick_size=7.0, label_size=8.0, title_size=9.3)
    plt.xticks(rotation=12, ha='right', fontsize=6.6)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_region_shap(out):
    # 권역별 SHAP 상위 5개 — v8_year_region_shap_top.csv 전체 기간 집계 대신
    # 각 권역의 연도별 mean을 변수별 집계
    d = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_shap_top.csv'))
    agg = d.groupby(['region', 'feature'])['mean_abs_shap'].sum().reset_index()
    top5 = {}
    for r in ['강남3구', '비강남']:
        sub = agg[agg['region'] == r].sort_values('mean_abs_shap', ascending=False).head(5)
        top5[r] = sub.set_index('feature')['mean_abs_shap']
    all_feats = sorted(set(top5['강남3구'].index) | set(top5['비강남'].index))
    gn = [top5['강남3구'].get(f, 0) for f in all_feats]
    bn = [top5['비강남'].get(f, 0) for f in all_feats]
    order = sorted(range(len(all_feats)), key=lambda i: max(gn[i], bn[i]), reverse=True)
    all_feats = [kr(all_feats[i]) for i in order]
    gn = [gn[i] for i in order]
    bn = [bn[i] for i in order]
    x = np.arange(len(all_feats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar(x - w/2, gn, w, label='강남3구', color='#d62728')
    ax.bar(x + w/2, bn, w, label='비강남', color='#1f77b4')
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_feature_label(v) for v in all_feats], rotation=0, ha='center', fontsize=6.6)
    ax.set_ylabel('연도 합산 평균 |SHAP|')
    ax.set_title('권역별 SHAP 주요 변수 비교')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.01,
        0.96,
        '강남3구: 상업·유동인구 신호 / 비강남: 주거·교육·물리 속성 신호',
        transform=ax.transAxes,
        fontsize=6.8,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
    )
    tune_axes(ax, tick_size=7.0, label_size=8.0, title_size=9.3)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_year_region_heatmap(out):
    s = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_summary.csv'))
    pv = s.pivot(index='year', columns='region', values='r2')
    pv = pv[['강남3구', '비강남', '전체']]
    fig, ax = plt.subplots(figsize=(4.9, 3.7))
    im = ax.imshow(pv.values, cmap='RdYlGn', vmin=0.85, vmax=0.97, aspect='auto')
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_title('연도×권역 XGBoost R² 히트맵')
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='white' if v < 0.9 else 'black', fontsize=7.8)
    if 2022 in list(pv.index):
        row = list(pv.index).index(2022)
        ax.add_patch(Rectangle((-0.5, row - 0.5), len(pv.columns), 1,
                               fill=False, edgecolor='#DC2626', linewidth=2.0))
        ax.text(len(pv.columns) - 0.5, row - 0.58, '2022 조정기 동시 저하',
                ha='right', va='bottom', fontsize=6.7, color='#991B1B')
    tune_axes(ax, tick_size=7.5, label_size=8.0, title_size=9.3)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_top1_timeline(out):
    d = pd.read_csv(os.path.join(ROOT, 'results/v8_year_region_shap_top.csv'))
    top1 = d[d['rank'] == 1].copy()
    pv = top1.pivot(index='year', columns='region', values='feature')[['강남3구', '비강남']]
    pv.to_csv(os.path.join(ROOT, 'results/v8_top1_by_year_region.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 2.85))
    years = pv.index.astype(int).tolist()
    ax.plot(years, [1] * len(years), 'o', color='#d62728', markersize=7.5)
    ax.plot(years, [0] * len(years), 'o', color='#1f77b4', markersize=7.5)
    for y, f in zip(years, pv['강남3구']):
        ha = 'right' if y == max(years) else 'left' if y == min(years) else 'center'
        ax.annotate(wrap_feature_label(kr(f)), (y, 1), textcoords="offset points", xytext=(0, 10),
                    ha=ha, fontsize=5.7, color='#d62728', clip_on=False)
    for y, f in zip(years, pv['비강남']):
        ha = 'right' if y == max(years) else 'left' if y == min(years) else 'center'
        ax.annotate(wrap_feature_label(kr(f)), (y, 0), textcoords="offset points", xytext=(0, -24),
                    ha=ha, fontsize=5.7, color='#1f77b4', clip_on=False)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['비강남', '강남3구'])
    ax.set_xticks(years)
    ax.set_xlim(min(years) - 0.45, max(years) + 0.45)
    ax.set_ylim(-0.62, 1.55)
    ax.set_title('연도별 최상위 SHAP 변수 변화: 강남3구 vs 비강남')
    ax.grid(True, axis='x', alpha=0.3)
    ax.text(
        0.01,
        0.95,
        '강남3구는 상위 신호가 국면별로 교체, 비강남은 건물연령 중심으로 안정',
        transform=ax.transAxes,
        fontsize=6.2,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
    )
    tune_axes(ax, tick_size=7.0, label_size=8.0, title_size=9.0)
    plt.tight_layout(pad=0.9)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def main():
    print("v8 데이터 로딩...")
    df = load_v8()
    print(f"데이터: {len(df):,}")
    print("XGBoost 적합...")
    model, tr, te = fit_xgb(df, TREE_FEATS)

    sample = te.sample(min(5000, len(te)), random_state=42)
    print("SHAP 계산...")
    sv = shap_values(model, sample[TREE_FEATS])

    print("그림 생성...")
    fig_shap_bar(sv, TREE_FEATS, os.path.join(FIGS, 'fig4_shap_bar.png'))
    fig_shap_summary(sv, sample, TREE_FEATS, os.path.join(FIGS, 'fig5_shap_summary.png'))
    fig_dependence(sv, sample, TREE_FEATS, '건물연령',
                   os.path.join(FIGS, 'fig6_dep_건물연령.png'),
                   'SHAP 의존도: 건물연령 U자형 비선형 패턴')
    fig_dependence(sv, sample, TREE_FEATS, 'childcare_count_1000m',
                   os.path.join(FIGS, 'fig7_dep_childcare_count_1000m.png'),
                   'SHAP 의존도: 어린이집 1km 내 개수')
    fig_dependence(sv, sample, TREE_FEATS, 'subway_nearest_m',
                   os.path.join(FIGS, 'fig8_dep_subway_nearest.png'),
                   'SHAP 의존도: 지하철 최근접거리 임계반응형')
    fig_dependence(sv, sample, TREE_FEATS, 'department_nearest_m',
                   os.path.join(FIGS, 'fig9_dep_department_nearest.png'),
                   'SHAP 의존도: 백화점 최근접거리')

    fig_ablation(os.path.join(FIGS, 'fig10_ablation.png'))
    fig_region_shap(os.path.join(FIGS, 'fig11_region_shap.png'))
    fig_year_region_heatmap(os.path.join(FIGS, 'fig12_year_region_heatmap.png'))
    fig_top1_timeline(os.path.join(FIGS, 'fig13_top1_timeline.png'))
    print(f"저장 완료: figures/fig4~13")


if __name__ == '__main__':
    main()
