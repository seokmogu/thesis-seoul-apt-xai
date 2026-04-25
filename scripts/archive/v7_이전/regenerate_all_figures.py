#!/usr/bin/env python3
"""
논문 그림 전체 재생성 — 한글 폰트 + 학술 논문 품질 + 가독성 개선
- fig1: 연구 흐름도 (matplotlib로 재제작, 흰색 배경)
- fig2: XGBoost 개념도 (matplotlib로 재제작, 흰색 배경)
- fig3: SHAP 프레임워크 (matplotlib로 재제작)
- fig4~fig10: SHAP 그래프 (한글 폰트 적용, 축 레이블 개선)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import pandas as pd
import json
import shap
import os
import pickle

# === 폰트 설정 ===
font_path = '/usr/share/fonts/truetype/nanum/NanumGothicLight.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 학술 논문 색상 팔레트 (절제된 3색)
COLOR_PRIMARY = '#1f77b4'    # 진한 파랑
COLOR_SECONDARY = '#ff7f0e'  # 주황
COLOR_ACCENT = '#2ca02c'     # 초록
COLOR_LIGHT = '#e8e8e8'      # 연한 회색
COLOR_GANGNAM = '#d62728'    # 빨강
COLOR_NON_GANGNAM = '#1f77b4'  # 파랑


def save_fig(fig, name, subdir='figures'):
    """그림 저장 (PNG 300dpi)"""
    out_dir = os.path.join(BASE_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✅ {path} ({os.path.getsize(path)//1024}KB)")


# ============================================================
# 그림 1: 연구 흐름도 (Research Framework)
# ============================================================
def make_fig1():
    print("📊 그림 1: 연구 흐름도...")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 스타일 정의
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333333', linewidth=1.5)
    title_props = dict(fontsize=12, fontweight='bold', ha='center', va='center')
    content_props = dict(fontsize=9, ha='center', va='center', linespacing=1.5)
    arrow_props = dict(arrowstyle='->', color='#333333', lw=1.5)
    
    # 단계별 박스
    stages = [
        (13.0, '제1단계: 데이터 수집', 
         '서울시 아파트 실거래가 (2019.01~2025.05)\n391,826건 · 215개 행정동\n국토교통부, 서울열린데이터, NEIS, 한국은행', '#E3F2FD'),
        (11.0, '제2단계: 데이터 전처리',
         '법정동→행정동 매핑 (Nominatim + GeoJSON)\n환경변수 구축 (18개 독립변수)\n학습(70%) / 검증(10%) / 테스트(20%) 분할', '#E8F5E9'),
        (9.0, '제3단계: 모형 구축',
         'OLS 회귀분석 (베이스라인)\nRandom Forest (배깅 앙상블)\nXGBoost (부스팅 앙상블) ← GridSearchCV 최적화', '#FFF3E0'),
        (7.0, '제4단계: 모형 평가',
         'R², RMSE, MAE, MAPE\n5-Fold 교차검증 (CV R² = 0.966)\n시간순 분할 강건성 점검', '#F3E5F5'),
        (5.0, '제5단계: XAI 분석 (SHAP)',
         'TreeSHAP 기반 변수 중요도 도출\nDependence Plot (비선형 효과)\nWaterfall Plot (개별 예측 해석)', '#E0F7FA'),
        (3.0, '제6단계: 지역별 비교분석',
         '강남3구 vs 비강남 SHAP 비교\n학원수·어린이집수의 비강남 우위 발견\n지역 특성에 따른 가격 결정요인 차이', '#FBE9E7'),
    ]
    
    for y, title, content, color in stages:
        # 배경 박스
        rect = mpatches.FancyBboxPatch((1, y-0.8), 8, 1.6, 
                                        boxstyle='round,pad=0.15',
                                        facecolor=color, edgecolor='#555555', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(5, y+0.5, title, fontsize=11, fontweight='bold', ha='center', va='center', color='#333333')
        ax.text(5, y-0.15, content, fontsize=8.5, ha='center', va='center', color='#444444', linespacing=1.4)
        
    # 화살표
    for i in range(len(stages)-1):
        y_from = stages[i][0] - 0.8
        y_to = stages[i+1][0] + 0.8
        ax.annotate('', xy=(5, y_to), xytext=(5, y_from),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
    
    # 결론 박스
    rect = mpatches.FancyBboxPatch((2, 0.8), 6, 1.0,
                                    boxstyle='round,pad=0.15',
                                    facecolor='#ECEFF1', edgecolor='#333333', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 1.3, '결론: 정책적 시사점 및 학술적 기여', 
            fontsize=11, fontweight='bold', ha='center', va='center', color='#333333')
    ax.annotate('', xy=(5, 1.8), xytext=(5, stages[-1][0]-0.8),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
    
    save_fig(fig, 'fig1_research_flow.png')


# ============================================================
# 그림 2: XGBoost 알고리즘 개념도
# ============================================================
def make_fig2():
    print("📊 그림 2: XGBoost 개념도...")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 학습 데이터
    rect = mpatches.FancyBboxPatch((0.3, 2.5), 2.2, 2, boxstyle='round,pad=0.2',
                                    facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.4, 4.0, '학습 데이터', fontsize=10, fontweight='bold', ha='center', color='#1565C0')
    ax.text(1.4, 3.3, 'X: 18개 독립변수\nY: 실거래가(만원)\n274,278건', fontsize=8, ha='center', va='center', color='#333')
    
    # Tree 1
    for i, (x_pos, label, sublabel) in enumerate([
        (3.8, 'Tree 1', '초기 모형'),
        (5.8, 'Tree 2', '잔차 학습'),
        (8.2, 'Tree N', '최종 보정'),
    ]):
        color = '#FFF3E0' if i < 2 else '#E8F5E9'
        edge_color = '#E65100' if i < 2 else '#2E7D32'
        rect = mpatches.FancyBboxPatch((x_pos-0.7, 2.8), 1.4, 1.5, boxstyle='round,pad=0.15',
                                        facecolor=color, edgecolor=edge_color, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x_pos, 3.9, label, fontsize=9, fontweight='bold', ha='center', color=edge_color)
        ax.text(x_pos, 3.3, sublabel, fontsize=7.5, ha='center', color='#555')
    
    # ... 생략 기호
    ax.text(7.0, 3.55, '· · ·', fontsize=16, ha='center', va='center', color='#888')
    
    # 화살표
    ax.annotate('', xy=(3.1, 3.55), xytext=(2.5, 3.55),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))
    ax.annotate('', xy=(5.1, 3.55), xytext=(4.5, 3.55),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))
    
    # 잔차 피드백
    ax.annotate('잔차\n(residuals)', xy=(4.5, 2.8), xytext=(4.5, 1.8),
                fontsize=8, ha='center', color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1, linestyle='--'))
    ax.annotate('잔차 학습', xy=(6.5, 2.8), xytext=(6.5, 1.8),
                fontsize=8, ha='center', color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1, linestyle='--'))
    
    # 가중합산
    rect = mpatches.FancyBboxPatch((9.5, 2.8), 2.2, 1.5, boxstyle='round,pad=0.15',
                                    facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(10.6, 3.9, '가중 합산', fontsize=10, fontweight='bold', ha='center', color='#6A1B9A')
    ax.text(10.6, 3.2, r'$\hat{y} = \sum \eta \cdot f_k(x)$' + '\n' + r'$\eta$: 학습률(0.1)', fontsize=8, ha='center', color='#333')
    
    ax.annotate('', xy=(9.5, 3.55), xytext=(8.9, 3.55),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))
    
    # 목적함수
    rect = mpatches.FancyBboxPatch((3.5, 5.2), 5, 1.2, boxstyle='round,pad=0.15',
                                    facecolor='#ECEFF1', edgecolor='#37474F', linewidth=1.3)
    ax.add_patch(rect)
    ax.text(6.0, 5.95, r'목적함수: $Obj = \sum L(y_i, \hat{y}_i) + \sum \Omega(f_k)$', 
            fontsize=9, fontweight='bold', ha='center', color='#37474F')
    ax.text(6.0, 5.45, '손실함수(예측 정확도) + 정규화(모형 복잡도 제어)', 
            fontsize=8, ha='center', color='#555')
    
    # 핵심 특징
    features = [
        ('정규화 (L1, L2)', 0.5),
        ('축소(Shrinkage)', 3.5),
        ('컬럼 서브샘플링', 6.5),
        ('Early Stopping', 9.5),
    ]
    for label, x_pos in features:
        rect = mpatches.FancyBboxPatch((x_pos, 0.3), 2.5, 0.8, boxstyle='round,pad=0.1',
                                        facecolor='#E8EAF6', edgecolor='#3949AB', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos+1.25, 0.7, label, fontsize=8, ha='center', va='center', color='#283593')
    
    save_fig(fig, 'fig2_xgboost_concept.png')


# ============================================================
# 그림 3: SHAP 분석 프레임워크
# ============================================================
def make_fig3():
    print("📊 그림 3: SHAP 프레임워크...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 1단계: 게임이론 기초
    rect = mpatches.FancyBboxPatch((1, 8.5), 8, 1.2, boxstyle='round,pad=0.15',
                                    facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5, 9.4, '게임이론 기반 (Shapley Value)', fontsize=11, fontweight='bold', ha='center', color='#1565C0')
    ax.text(5, 8.85, r'$\phi_i = \sum \frac{|S|!(p-|S|-1)!}{p!} [f(S \cup \{i\}) - f(S)]$', 
            fontsize=10, ha='center', color='#333')
    
    # 화살표
    ax.annotate('', xy=(5, 7.3), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    
    # 2단계: TreeSHAP
    rect = mpatches.FancyBboxPatch((1, 5.8), 8, 1.5, boxstyle='round,pad=0.15',
                                    facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5, 6.9, 'TreeSHAP (Lundberg et al., 2020)', fontsize=11, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(3.5, 6.2, '• 트리 기반 모형에 최적화\n• 다항시간 O(TLD²) 복잡도', 
            fontsize=8.5, ha='center', va='center', color='#333')
    ax.text(6.5, 6.2, '• 정확한 SHAP값 산출\n• XGBoost에 직접 적용', 
            fontsize=8.5, ha='center', va='center', color='#333')
    
    ax.annotate('', xy=(5, 4.5), xytext=(5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    
    # 입력
    rect = mpatches.FancyBboxPatch((0.5, 4.5), 3.5, 1.0, boxstyle='round,pad=0.1',
                                    facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(2.25, 5.15, '입력', fontsize=9, fontweight='bold', ha='center', color='#E65100')
    ax.text(2.25, 4.75, 'XGBoost 모형 + 테스트 5,000건', fontsize=8, ha='center', color='#333')
    
    # 3단계: 분석 결과
    outputs = [
        (0.8, '글로벌 해석', 'Bar Plot\nSummary Plot', '#E3F2FD', '#1565C0'),
        (3.8, '변수별 효과', 'Dependence\nPlot', '#E8F5E9', '#2E7D32'),
        (6.8, '개별 해석', 'Waterfall\nPlot', '#F3E5F5', '#6A1B9A'),
    ]
    
    for x, title, content, color, edge in outputs:
        rect = mpatches.FancyBboxPatch((x, 1.5), 2.5, 2.0, boxstyle='round,pad=0.15',
                                        facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+1.25, 3.1, title, fontsize=9, fontweight='bold', ha='center', color=edge)
        ax.text(x+1.25, 2.2, content, fontsize=8, ha='center', va='center', color='#444')
    
    # 화살표 (입력 → 출력들)
    for x in [2.05, 5.05, 8.05]:
        ax.annotate('', xy=(x, 3.5), xytext=(5, 4.5),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
    
    # 지역 비교 분석 (추가 박스)
    rect = mpatches.FancyBboxPatch((2.5, 0.2), 5, 0.9, boxstyle='round,pad=0.1',
                                    facecolor='#FBE9E7', edgecolor='#C62828', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(5, 0.65, '강남3구 vs 비강남 지역별 비교분석', fontsize=9, fontweight='bold', ha='center', color='#C62828')
    
    save_fig(fig, 'fig3_shap_framework.png')


# ============================================================
# 그림 4~10: SHAP 그래프 재생성 (한글 폰트)
# ============================================================
def regenerate_shap_figures():
    """SHAP 그래프 전체 재생성"""
    print("\n📊 SHAP 그래프 재생성 시작...")
    
    # 모델 및 데이터 로드
    model_path = os.path.join(RESULTS_DIR, 'xgb_model_v6.pkl')
    data_path = os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv')
    
    if not os.path.exists(model_path):
        print("  ⚠️ XGBoost 모델 파일 없음. 새로 학습합니다...")
        # 모델이 없으면 modeling 스크립트에서 SHAP 값만 필요
        # shap_values를 저장한 파일이 있는지 확인
        shap_path = os.path.join(RESULTS_DIR, 'shap_values_v6.pkl')
        if os.path.exists(shap_path):
            with open(shap_path, 'rb') as f:
                shap_data = pickle.load(f)
            print("  📂 SHAP 값 로드 완료")
            return shap_data
        else:
            print("  ❌ SHAP 값 파일도 없음. modeling_v6_dong.py 재실행 필요")
            return None
    
    import xgboost as xgb
    
    # 데이터 로드
    print("  📂 데이터 로드 중...")
    df = pd.read_csv(data_path)
    
    feature_cols = ['전용면적', '층', '건물연령', '강남구분', '지하철역수',
                    '초등학교수', '중학교수', '고등학교수', 'CCTV수', '백화점수',
                    '학원수', '어린이집수', '공원수', '도서관수',
                    '기준금리', 'CD금리', '소비자물가지수', 'M2']
    
    X = df[feature_cols]
    
    # 모델 로드
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print("  📂 모델 로드 완료")
    
    # SHAP 계산 (5000건 샘플)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    
    print("  ⏳ SHAP 값 계산 중 (5,000건)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # 저장
    shap_save = {'shap_values': shap_values, 'X_sample': X_sample, 'feature_cols': feature_cols}
    with open(os.path.join(RESULTS_DIR, 'shap_values_v6.pkl'), 'wb') as f:
        pickle.dump(shap_save, f)
    
    return shap_save


def make_shap_plots(shap_data):
    """SHAP 그래프 생성"""
    if shap_data is None:
        print("  ❌ SHAP 데이터 없음, 스킵")
        return
    
    shap_values = shap_data['shap_values']
    X_sample = shap_data['X_sample']
    feature_cols = shap_data['feature_cols']
    
    # 그림 4: Summary Plot (Beeswarm)
    print("📊 그림 4: SHAP Summary Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=18)
    plt.xlabel('SHAP 값 (모형 출력에 대한 영향, 만원)', fontsize=11)
    plt.title('')
    plt.tight_layout()
    save_fig(plt.gcf(), 'fig4_shap_summary.png')
    
    # 그림 5: Bar Plot
    print("📊 그림 5: SHAP Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=18, show=False)
    plt.xlabel('평균 |SHAP 값| (만원)', fontsize=11)
    plt.title('')
    plt.tight_layout()
    save_fig(plt.gcf(), 'fig5_shap_bar.png')
    
    # 그림 6: Dependence Plot — 전용면적
    print("📊 그림 6: 전용면적 Dependence Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('전용면적', shap_values.values, X_sample,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_xlabel('전용면적 (㎡)', fontsize=11)
    ax.set_ylabel('전용면적에 대한 SHAP 값 (만원)', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig(fig, 'fig6_dep_전용면적.png')
    
    # 그림 7: Dependence Plot — 건물연령
    print("📊 그림 7: 건물연령 Dependence Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('건물연령', shap_values.values, X_sample,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_xlabel('건물연령 (년)', fontsize=11)
    ax.set_ylabel('건물연령에 대한 SHAP 값 (만원)', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig(fig, 'fig7_dep_건물연령.png')
    
    # 그림 8: Waterfall Plot
    print("📊 그림 8: Waterfall Plot...")
    # 노원구 월계1동 사례 찾기 (또는 첫 번째 샘플)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.tight_layout()
    save_fig(plt.gcf(), 'fig8_waterfall_plot.png')
    
    # 그림 9: 강남 vs 비강남 비교
    print("📊 그림 9: 강남/비강남 SHAP 비교...")
    gangnam_mask = X_sample['강남구분'].values == 1
    
    gangnam_mean = np.abs(shap_values.values[gangnam_mask]).mean(axis=0)
    non_gangnam_mean = np.abs(shap_values.values[~gangnam_mask]).mean(axis=0)
    
    # 상위 10개
    total_importance = (gangnam_mean + non_gangnam_mean) / 2
    top_idx = np.argsort(total_importance)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top_idx))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gangnam_mean[top_idx]/1000, width,
                   label='강남3구', color=COLOR_GANGNAM, edgecolor='white', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, non_gangnam_mean[top_idx]/1000, width,
                   label='비강남', color=COLOR_NON_GANGNAM, edgecolor='white', linewidth=0.5, 
                   alpha=0.85, hatch='//')
    
    ax.set_ylabel('평균 |SHAP 값| (×1,000 만원)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_cols[i] for i in top_idx], rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11, frameon=True, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 상위 3개에 수치 표기
    for i, bar in enumerate(bars1[:3]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color=COLOR_GANGNAM)
    for i, bar in enumerate(bars2[:3]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color=COLOR_NON_GANGNAM)
    
    plt.tight_layout()
    save_fig(fig, 'fig9_gangnam_comparison.png')


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("논문 그림 전체 재생성 (학술 논문 품질)")
    print("=" * 60)
    
    # 다이어그램 (matplotlib 직접 제작)
    make_fig1()
    make_fig2()
    make_fig3()
    
    # SHAP 그래프 (한글 폰트 + 가독성 개선)
    shap_data = regenerate_shap_figures()
    make_shap_plots(shap_data)
    
    print("\n" + "=" * 60)
    print("✅ 모든 그림 재생성 완료!")
    print(f"📁 저장 위치: {FIGURES_DIR}")
    print("=" * 60)
