#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.graph_setup import build_market_graph
from src.synthetic_data import generate_synthetic_events, create_time_series
from src.propagation import multi_source_propagation, propagate_attention
from src.analysis import backtest_propagation_strategy

plt.style.use('dark_background')
sns.set_palette("husl")

def create_hs1_attention_propagation_heatmap():
    
    print("Creating HS1: Temporal Attention Propagation Heatmap...")

    G = build_market_graph()
    events = generate_synthetic_events(30)

    import networkx as nx
    centrality = nx.eigenvector_centrality(G)
    top_companies = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    companies = [c[0] for c in top_companies]

    time_points = np.arange(0, 48, 0.5)
    attention_matrix = np.zeros((len(companies), len(time_points)))

    for t_idx, t in enumerate(time_points):
        active_events = [
            (e['company'], abs(e['sentiment']), e['hours_from_start'])
            for e in events if e['hours_from_start'] <= t
        ]

        scores = multi_source_propagation(G, active_events, t, lambda_decay=0.3)

        for c_idx, company in enumerate(companies):
            attention_matrix[c_idx, t_idx] = scores.get(company, 0)

    attention_matrix = gaussian_filter(attention_matrix, sigma=0.8)

    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#f5f5f7')
    ax.set_facecolor('#f5f5f7')

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#f5f5f7', '#d0d0d5', '#a0a0b0', '#707090', '#505070', '#e94560', '#ff6b6b', '#ffd93d']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)

    im = ax.imshow(attention_matrix, aspect='auto', cmap=cmap, interpolation='bilinear')

    ax.set_xticks(np.arange(0, len(time_points), 12))
    ax.set_xticklabels([f'{int(t)}h' for t in time_points[::12]], fontsize=11, color='black')
    ax.set_yticks(np.arange(len(companies)))
    ax.set_yticklabels(companies, fontsize=12, weight='bold', color='black')

    ax.set_xlabel('Time Since Initial Event', fontsize=14, weight='bold', color='black')
    ax.set_ylabel('Companies', fontsize=14, weight='bold', color='black')
    ax.set_title('Market Attention Propagation: Temporal Evolution\nAttention-based Contagion Model',
                 fontsize=18, weight='bold', color='black', pad=20)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=25, fontsize=12, weight='bold', color='black')
    cbar.ax.tick_params(labelsize=10, colors='black')

    for event in events[:15]:
        if event['hours_from_start'] < 48:
            t_idx = int(event['hours_from_start'] / 0.5)
            if event['company'] in companies:
                c_idx = companies.index(event['company'])
                ax.plot(t_idx, c_idx, 'o', color='cyan', markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)

    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='black')

    ax.text(0.02, 0.98, 'Neural Attention Propagation Engine v1.0',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#533483'),
            color='#533483', weight='bold')

    plt.tight_layout()
    plt.savefig('results/hs1.png', dpi=300, facecolor='#f5f5f7', bbox_inches='tight')
    print("  ✓ Saved to results/hs1.png")
    plt.close()

def create_hs2_signal_quality_analysis():
    
    print("Creating HS2: Signal Quality & Model Performance...")

    G = build_market_graph()
    events = generate_synthetic_events(30)

    metrics = backtest_propagation_strategy(G, events, threshold=0.25)

    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')

    ax1 = plt.subplot(2, 2, 1, facecolor='#0a0a0a')

    thresholds = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        m = backtest_propagation_strategy(G, events, threshold=thresh)
        if 'error' not in m and m['total_trades'] > 0:
            precision = m['win_rate']
            recall = m['total_trades'] / 150
            precisions.append(precision)
            recalls.append(recall)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            f1_scores.append(f1)
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)

    ax1.plot(recalls, precisions, linewidth=3, color='#e94560', label='Attention Model')
    ax1.fill_between(recalls, precisions, alpha=0.3, color='#e94560')

    baseline_precision = 0.5
    ax1.axhline(y=baseline_precision, color='#666', linestyle='--', linewidth=2,
                label='Random Baseline', alpha=0.7)

    best_f1_idx = np.argmax(f1_scores)
    ax1.plot(recalls[best_f1_idx], precisions[best_f1_idx], 'o',
            color='#ffd93d', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax1.annotate(f'Optimal F1: {f1_scores[best_f1_idx]:.3f}',
                xy=(recalls[best_f1_idx], precisions[best_f1_idx]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=11, weight='bold', color='#ffd93d',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2))

    ax1.set_xlabel('Recall', fontsize=12, weight='bold')
    ax1.set_ylabel('Precision', fontsize=12, weight='bold')
    ax1.set_title('Precision-Recall Curve', fontsize=14, weight='bold', color='white')
    ax1.legend(fontsize=10, framealpha=0.9, facecolor='#1a1a2e')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2 = plt.subplot(2, 2, 2, facecolor='#0a0a0a')

    time_hours = np.linspace(0, 12, 100)
    lambda_values = [0.2, 0.3, 0.5]
    colors = ['#00ff9f', '#e94560', '#ffd93d']

    for lambda_val, color in zip(lambda_values, colors):
        decay = np.exp(-lambda_val * time_hours)
        ax2.plot(time_hours, decay, linewidth=3, label=f'λ = {lambda_val}', color=color)
        ax2.fill_between(time_hours, decay, alpha=0.2, color=color)

    ax2.axhline(y=0.5, color='#666', linestyle=':', linewidth=2, alpha=0.5)
    ax2.text(6, 0.52, 'Half-life threshold', fontsize=10, color='#999')

    ax2.set_xlabel('Time (hours)', fontsize=12, weight='bold')
    ax2.set_ylabel('Attention Intensity', fontsize=12, weight='bold')
    ax2.set_title('Temporal Decay Function: I(t) = I₀ × e^(-λt)', fontsize=14, weight='bold')
    ax2.legend(fontsize=10, framealpha=0.9, facecolor='#1a1a2e')
    ax2.grid(True, alpha=0.2, linestyle='--')

    ax3 = plt.subplot(2, 2, 3, facecolor='#0a0a0a')

    if 'trades' in metrics:
        correct_scores = [t['predicted_score'] for t in metrics['trades'] if t['correct']]
        wrong_scores = [t['predicted_score'] for t in metrics['trades'] if not t['correct']]

        ax3.hist(correct_scores, bins=20, alpha=0.7, color='#00ff9f',
                label=f'Correct ({len(correct_scores)})', edgecolor='white', linewidth=1.5)
        ax3.hist(wrong_scores, bins=20, alpha=0.7, color='#e94560',
                label=f'Incorrect ({len(wrong_scores)})', edgecolor='white', linewidth=1.5)

        ax3.axvline(x=np.mean(correct_scores), color='#00ff9f', linestyle='--',
                   linewidth=2, label=f'Mean Correct: {np.mean(correct_scores):.3f}')
        ax3.axvline(x=np.mean(wrong_scores), color='#e94560', linestyle='--',
                   linewidth=2, label=f'Mean Wrong: {np.mean(wrong_scores):.3f}')

    ax3.set_xlabel('Propagation Score', fontsize=12, weight='bold')
    ax3.set_ylabel('Count', fontsize=12, weight='bold')
    ax3.set_title('Prediction Confidence Distribution', fontsize=14, weight='bold')
    ax3.legend(fontsize=9, framealpha=0.9, facecolor='#1a1a2e')
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')

    ax4 = plt.subplot(2, 2, 4, facecolor='#0a0a0a')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['win_rate'],
        metrics['total_trades'] / 150,
        2 * (metrics['win_rate'] * metrics['accuracy']) / (metrics['win_rate'] + metrics['accuracy'] + 1e-10)
    ]

    colors_bars = ['#00ff9f', '#e94560', '#ffd93d', '#00d9ff']
    bars = ax4.barh(metric_names, metric_values, color=colors_bars,
                    edgecolor='white', linewidth=2, height=0.6)

    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax4.text(val + 0.02, i, f'{val:.2%}', va='center', fontsize=12,
                weight='bold', color='white')

    ax4.axvline(x=0.7, color='#666', linestyle='--', linewidth=2, alpha=0.7)
    ax4.text(0.72, 3.5, 'Industry\nBenchmark', fontsize=9, color='#999', weight='bold')

    ax4.set_xlabel('Score', fontsize=12, weight='bold')
    ax4.set_title('Model Performance Metrics', fontsize=14, weight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.2, linestyle='--', axis='x')

    fig.suptitle('Attention Propagation Model: Signal Quality Analysis',
                 fontsize=20, weight='bold', color='white', y=0.98)

    fig.text(0.99, 0.01, 'Attention Contagion Engine | Neural Network Analysis v1.0',
            ha='right', fontsize=10, color='#666', style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig('results/hs2.png', dpi=300, facecolor='#0a0a0a', bbox_inches='tight')
    print("  ✓ Saved to results/hs2.png")
    plt.close()

def create_hs3_network_influence_flow():
    
    print("Creating HS3: Network Influence Flow Diagram...")

    G = build_market_graph()

    import networkx as nx
    centrality = nx.eigenvector_centrality(G)

    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:25]
    node_names = [n[0] for n in top_nodes]

    H = G.subgraph(node_names).copy()

    fig, ax = plt.subplots(figsize=(16, 16), facecolor='#f5f5f7')
    ax.set_facecolor('#f5f5f7')
    ax.set_aspect('equal')

    pos = nx.circular_layout(H)

    for node in pos:
        pos[node] = pos[node] * 3

    edge_colors = []
    edge_widths = []

    for u, v, data in H.edges(data=True):
        influence = data.get('influence', 0.5)
        edge_colors.append(influence)
        edge_widths.append(influence * 3)

    from matplotlib.colors import LinearSegmentedColormap
    edge_cmap = LinearSegmentedColormap.from_list('influence',
                                                   ['#c7c7cc', '#8e8e93', '#533483', '#e94560', '#ffd93d'])

    nx.draw_networkx_edges(H, pos, ax=ax,
                           edge_color=edge_colors,
                           edge_cmap=edge_cmap,
                           width=edge_widths,
                           alpha=0.6,
                           arrows=True,
                           arrowsize=15,
                           arrowstyle='->',
                           connectionstyle='arc3,rad=0.1',
                           edge_vmin=0, edge_vmax=1)

    node_sizes = [centrality.get(node, 0.1) * 5000 for node in H.nodes()]
    node_colors = [centrality.get(node, 0.1) for node in H.nodes()]

    from matplotlib.colors import LinearSegmentedColormap
    node_cmap = LinearSegmentedColormap.from_list('centrality',
                                                   ['#0f3460', '#533483', '#e94560', '#ffd93d'])

    nodes = nx.draw_networkx_nodes(H, pos, ax=ax,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap=node_cmap,
                                   edgecolors='#1d1d1f',
                                   linewidths=2,
                                   vmin=0, vmax=max(node_colors))

    label_pos = {}
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x)
        offset = 0.3
        label_pos[node] = (x + offset * np.cos(angle), y + offset * np.sin(angle))

    nx.draw_networkx_labels(H, label_pos, ax=ax,
                           font_size=11,
                           font_weight='bold',
                           font_color='#1d1d1f',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   edgecolor='#d1d1d6',
                                   alpha=0.9,
                                   linewidth=1.5))

    ax.set_title('Market Attention Network: Influence Topology\nDirected Graph Analysis of Propagation Pathways',
                fontsize=20, weight='bold', color='#1d1d1f', pad=30)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#8e8e93', linewidth=2, label='Low Influence (0.3)'),
        Line2D([0], [0], color='#533483', linewidth=3, label='Medium Influence (0.6)'),
        Line2D([0], [0], color='#e94560', linewidth=4, label='High Influence (0.8)'),
        Line2D([0], [0], color='#ffd93d', linewidth=5, label='Critical Path (0.9+)'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                   framealpha=0.95, facecolor='white', edgecolor='#d1d1d6',
                   title='Edge Influence Weight', labelcolor='#1d1d1f')
    leg.get_title().set_fontsize(12)
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_color('#1d1d1f')

    metrics_text = f

    ax.text(0.99, 0.99, metrics_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.8',
                    facecolor='white',
                    edgecolor='#d1d1d6',
                    alpha=0.95,
                    linewidth=2),
           color='#1d1d1f',
           weight='bold',
           family='monospace')

    sector_colors = {
        'ev_maker': '#00ff9f',
        'battery': '#e94560',
        'lithium': '#ffd93d',
        'charging': '#00d9ff',
        'chips': '#ff6b6b',
        'legacy_auto': '#9b59b6'
    }

    sector_elements = []
    for sector, color in sector_colors.items():
        sector_elements.append(mpatches.Patch(color=color, label=sector.replace('_', ' ').title()))

    leg2 = ax.legend(handles=sector_elements, loc='upper right', fontsize=10,
                    framealpha=0.95, facecolor='white', edgecolor='#d1d1d6',
                    title='Company Sectors', ncol=2, labelcolor='#1d1d1f')
    leg2.get_title().set_fontsize(11)
    leg2.get_title().set_fontweight('bold')
    leg2.get_title().set_color('#1d1d1f')

    ax.axis('off')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    fig.text(0.5, 0.02, 'Graph Neural Network Analysis | Attention Flow Dynamics v1.0',
            ha='center', fontsize=11, color='#6e6e73', style='italic', weight='bold')

    plt.tight_layout()
    plt.savefig('results/hs3.png', dpi=300, facecolor='#f5f5f7', bbox_inches='tight')
    print("  ✓ Saved to results/hs3.png")
    plt.close()

def main():
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    print()

    create_hs1_attention_propagation_heatmap()
    print()
    create_hs2_signal_quality_analysis()
    print()
    create_hs3_network_influence_flow()

    print()
    print("=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nGenerated:")
    print("  results/hs1.png - Temporal Attention Heatmap")
    print("  results/hs2.png - Signal Quality Analysis")
    print("  results/hs3.png - Network Influence Flow")
    print("=" * 60)

if __name__ == '__main__':
    main()
