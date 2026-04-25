#!/usr/bin/env python3
"""Publication-quality thesis diagrams using matplotlib — all English text"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG_DIR = '/home/nexus/thesis-seoul-apt-xai/figures'

# Colors: restrained 3-color palette + grays
C_PRIMARY = '#2C5F8A'    # Deep blue
C_SECONDARY = '#D4853A'  # Warm orange  
C_ACCENT = '#5B8C5A'     # Muted green
C_LIGHT1 = '#E8EFF5'     # Light blue bg
C_LIGHT2 = '#FDF0E2'     # Light orange bg
C_LIGHT3 = '#E5F0E4'     # Light green bg
C_LIGHT4 = '#F0E5F0'     # Light purple bg
C_LIGHT5 = '#FDE8E8'     # Light red bg
C_GRAY = '#F5F5F5'
C_BORDER = '#333333'
FONT = {'family': 'sans-serif', 'size': 10}

def draw_box(ax, x, y, w, h, text, fill='white', border='#333', lw=1.2, fontsize=9, bold=False, align='center'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fill, edgecolor=border, linewidth=lw)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            fontfamily='sans-serif')

def draw_arrow(ax, x1, y1, x2, y2, color='#666'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def fig1_research_framework():
    """Research Framework Flowchart"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Stage 1: Data Collection
    draw_box(ax, 0.5, 8.8, 6, 0.5, 'Stage 1: Data Collection (2019.01 - 2025.12)',
             fill=C_LIGHT1, border=C_PRIMARY, fontsize=11, bold=True, lw=2)
    draw_box(ax, 0.7, 7.8, 1.7, 0.8,
             'Apartment\nTransactions\n391,826 records\n(MOLIT API)',
             fill='white', border=C_PRIMARY, fontsize=7.5)
    draw_box(ax, 2.65, 7.8, 1.7, 0.8,
             'Infrastructure\nSchools, Subway,\nCCTV, Dept Stores\n(NEIS, Seoul API)',
             fill='white', border=C_PRIMARY, fontsize=7.5)
    draw_box(ax, 4.6, 7.8, 1.7, 0.8,
             'Macroeconomic\nBase Rate, CD Rate,\nCPI, M2\n(BOK ECOS)',
             fill='white', border=C_PRIMARY, fontsize=7.5)
    
    # Arrow
    draw_arrow(ax, 3.5, 7.7, 3.5, 7.0)
    
    # Stage 2: Preprocessing
    draw_box(ax, 0.5, 6.2, 6, 0.5, 'Stage 2: Preprocessing & Dong-Level Integration',
             fill=C_LIGHT3, border=C_ACCENT, fontsize=11, bold=True, lw=2)
    draw_box(ax, 0.7, 5.3, 1.7, 0.7,
             'Dong Mapping\nNominatim +\nGeoJSON Spatial Join\n215 Admin. Dongs',
             fill='white', border=C_ACCENT, fontsize=7.5)
    draw_box(ax, 2.65, 5.3, 1.7, 0.7,
             'Variable Merge\n14 Independent Vars\n+ Gangnam Dummy',
             fill='white', border=C_ACCENT, fontsize=7.5)
    draw_box(ax, 4.6, 5.3, 1.7, 0.7,
             'Data Split\nTrain 70%\nVal 10% / Test 20%',
             fill='white', border=C_ACCENT, fontsize=7.5)
    
    # Arrow
    draw_arrow(ax, 3.5, 5.2, 3.5, 4.5)
    
    # Stage 3: Modeling
    draw_box(ax, 0.5, 3.7, 6, 0.5, 'Stage 3: Model Comparison',
             fill=C_LIGHT2, border=C_SECONDARY, fontsize=11, bold=True, lw=2)
    draw_box(ax, 0.7, 2.9, 1.7, 0.6,
             'OLS Regression\nR² = 0.587',
             fill='white', border=C_SECONDARY, fontsize=8.5)
    draw_box(ax, 2.65, 2.9, 1.7, 0.6,
             'Random Forest\nR² = 0.956',
             fill='white', border=C_SECONDARY, fontsize=8.5)
    draw_box(ax, 4.6, 2.9, 1.7, 0.6,
             'XGBoost\nR² = 0.967 ★',
             fill=C_LIGHT2, border=C_SECONDARY, fontsize=9, bold=True, lw=2.5)
    
    # Arrow
    draw_arrow(ax, 3.5, 2.8, 3.5, 2.1)
    
    # Stage 4: XAI
    draw_box(ax, 0.5, 1.3, 6, 0.5, 'Stage 4: Explainable AI (SHAP Analysis)',
             fill=C_LIGHT4, border='#6A1B9A', fontsize=11, bold=True, lw=2)
    draw_box(ax, 0.7, 0.5, 1.7, 0.6,
             'Global Interpretation\nFeature Importance\nSummary Plot',
             fill='white', border='#6A1B9A', fontsize=7.5)
    draw_box(ax, 2.65, 0.5, 1.7, 0.6,
             'Local Interpretation\nForce Plot\nIndividual Prediction',
             fill='white', border='#6A1B9A', fontsize=7.5)
    draw_box(ax, 4.6, 0.5, 1.7, 0.6,
             'Regional Analysis\nGangnam vs\nNon-Gangnam',
             fill='white', border='#6A1B9A', fontsize=7.5)
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_research_framework.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  fig1_research_framework.png")

def fig2_xgboost_architecture():
    """XGBoost Architecture Diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    
    # Input
    draw_box(ax, 0.1, 1.5, 1.5, 1.5,
             'Training Data\n\nX: 14 features\ny: Price\n(391,826 obs)',
             fill=C_LIGHT1, border=C_PRIMARY, fontsize=8)
    
    # Trees
    draw_arrow(ax, 1.7, 2.25, 2.1, 2.25)
    
    for i, (x, label, sub) in enumerate([
        (2.2, 'Tree 1', 'Base'),
        (3.4, 'Tree 2', 'Residual 1'),
        (5.2, 'Tree N', 'Residual N-1'),
    ]):
        fill = C_LIGHT3 if i < 2 else C_LIGHT2
        draw_box(ax, x, 1.2, 1.0, 2.0, f'{label}\n\n{sub}',
                 fill=fill, border=C_ACCENT if i < 2 else C_SECONDARY, fontsize=8.5, bold=True)
    
    # Dots
    ax.text(4.7, 2.2, '···', fontsize=20, ha='center', va='center', color='#666')
    
    # Arrows between trees
    draw_arrow(ax, 3.3, 2.25, 3.4, 2.25)
    ax.text(3.35, 2.6, 'residuals', fontsize=6.5, ha='center', color='#888', style='italic')
    draw_arrow(ax, 4.5, 2.25, 4.65, 2.25)
    draw_arrow(ax, 4.85, 2.25, 5.2, 2.25)
    
    # Sum
    draw_arrow(ax, 6.3, 2.25, 6.6, 2.25)
    draw_box(ax, 6.7, 1.5, 1.0, 1.5,
             'Σ\n\nŷ = Σ η·fₖ(x)',
             fill=C_LIGHT2, border=C_SECONDARY, fontsize=8.5, bold=True)
    
    # Output
    draw_arrow(ax, 7.8, 2.25, 8.1, 2.25)
    draw_box(ax, 8.15, 1.7, 0.75, 1.0,
             'Predicted\nPrice\n(10K KRW)',
             fill=C_LIGHT3, border=C_ACCENT, fontsize=7.5, bold=True)
    
    # Objective function (below)
    draw_box(ax, 3.0, 0.1, 3.5, 0.8,
             'Objective:  L(θ) = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)    where  Ω(f) = γT + ½λ‖w‖²',
             fill=C_GRAY, border='#999', fontsize=8, lw=0.8)
    
    ax.annotate('', xy=(4.75, 0.95), xytext=(7.2, 1.4),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, ls='--'))
    ax.text(6.3, 1.0, 'optimize', fontsize=7, color='#888', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_xgboost_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  fig2_xgboost_architecture.png")

def fig3_shap_framework():
    """SHAP Framework Diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Game Theory
    draw_box(ax, 0.3, 5.7, 7.4, 0.5, 'Game Theory Foundation: Shapley Value',
             fill=C_LIGHT1, border=C_PRIMARY, fontsize=11, bold=True, lw=2)
    
    draw_box(ax, 0.5, 4.7, 3.3, 0.8,
             'Shapley Value Formula\nφᵢ = Σ |S|!(p-|S|-1)!/p!\n× [f(S∪{i}) - f(S)]',
             fill='white', border=C_PRIMARY, fontsize=8)
    
    draw_box(ax, 4.1, 4.7, 3.4, 0.8,
             'Axioms: Efficiency, Symmetry,\nDummy Player, Additivity\n→ Unique fair allocation',
             fill='white', border=C_PRIMARY, fontsize=8)
    
    # Arrow
    draw_arrow(ax, 4.0, 4.5, 4.0, 3.9)
    
    # TreeSHAP
    draw_box(ax, 1.5, 3.2, 5.0, 0.5, 'TreeSHAP: Optimized for Tree Ensembles',
             fill=C_LIGHT3, border=C_ACCENT, fontsize=11, bold=True, lw=2)
    
    draw_box(ax, 1.7, 2.3, 2.2, 0.7,
             'XGBoost Model\n(R² = 0.967, CV = 0.965)\n14 features, 215 dongs',
             fill='white', border=C_ACCENT, fontsize=7.5, bold=True)
    draw_box(ax, 4.1, 2.3, 2.2, 0.7,
             'Computational Efficiency\nO(TLD²) complexity\nExact SHAP values',
             fill='white', border=C_ACCENT, fontsize=7.5)
    
    # Arrow
    draw_arrow(ax, 4.0, 2.1, 4.0, 1.5)
    
    # SHAP Outputs
    draw_box(ax, 0.3, 0.7, 7.4, 0.5, 'SHAP Interpretation Outputs',
             fill=C_LIGHT2, border=C_SECONDARY, fontsize=11, bold=True, lw=2)
    
    draw_box(ax, 0.5, 0.05, 2.2, 0.5,
             'Global: Feature Importance,\nSummary & Dependence Plots',
             fill='white', border=C_SECONDARY, fontsize=7.5)
    draw_box(ax, 2.9, 0.05, 2.2, 0.5,
             'Local: Force Plot,\nIndividual Predictions',
             fill='white', border=C_SECONDARY, fontsize=7.5)
    draw_box(ax, 5.3, 0.05, 2.2, 0.5,
             'Regional: Gangnam vs\nNon-Gangnam Comparison',
             fill='white', border=C_SECONDARY, fontsize=7.5)
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_shap_framework.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  fig3_shap_framework.png")

if __name__ == '__main__':
    print("=== Generating Publication Diagrams ===")
    fig1_research_framework()
    fig2_xgboost_architecture()
    fig3_shap_framework()
    print("Done!")
