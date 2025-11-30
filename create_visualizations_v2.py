#!/usr/bin/env python3

import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def create_hs4_neon_network():
    
    print("Creating HS4: Neon Network - Attention Flow Topology...")

    G = nx.DiGraph()

    center = ['TSLA']
    ring1 = ['CATL', 'PANW', 'BYD']
    ring2 = ['ALB', 'SQM', 'LAC', 'LTHM']
    ring3 = ['FCX', 'SCCO', 'REDW', 'LI']

    all_nodes = center + ring1 + ring2 + ring3

    sectors = {
        'TSLA': 'Tech/Auto', 'CATL': 'Battery', 'PANW': 'Battery', 'BYD': 'Auto',
        'ALB': 'Lithium', 'SQM': 'Lithium', 'LAC': 'Lithium', 'LTHM': 'Lithium',
        'FCX': 'Copper', 'SCCO': 'Copper', 'REDW': 'Recycling', 'LI': 'Lithium'
    }

    for node in all_nodes:
        G.add_node(node, sector=sectors[node])

    edges = [
        ('TSLA', 'CATL', 0.95), ('TSLA', 'PANW', 0.90), ('TSLA', 'BYD', 0.75),
        ('TSLA', 'ALB', 0.85), ('TSLA', 'SQM', 0.75),
        ('CATL', 'ALB', 0.90), ('CATL', 'SQM', 0.85), ('CATL', 'LAC', 0.70),
        ('PANW', 'ALB', 0.80), ('PANW', 'LTHM', 0.75),
        ('BYD', 'CATL', 0.85), ('BYD', 'ALB', 0.80),
        ('ALB', 'FCX', 0.50), ('ALB', 'REDW', 0.60),
        ('SQM', 'FCX', 0.45), ('SQM', 'LI', 0.55),
        ('LAC', 'LI', 0.65), ('LTHM', 'REDW', 0.50),
        ('REDW', 'CATL', 0.40), ('FCX', 'TSLA', 0.35),
    ]

    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    pos = {}
    angles_ring1 = np.linspace(0, 2*np.pi, len(ring1), endpoint=False)
    angles_ring2 = np.linspace(0, 2*np.pi, len(ring2), endpoint=False) + np.pi/4
    angles_ring3 = np.linspace(0, 2*np.pi, len(ring3), endpoint=False)

    pos['TSLA'] = (0, 0)
    for i, node in enumerate(ring1):
        pos[node] = (1.5*np.cos(angles_ring1[i]), 1.5*np.sin(angles_ring1[i]))
    for i, node in enumerate(ring2):
        pos[node] = (3*np.cos(angles_ring2[i]), 3*np.sin(angles_ring2[i]))
    for i, node in enumerate(ring3):
        pos[node] = (4.5*np.cos(angles_ring3[i]), 4.5*np.sin(angles_ring3[i]))

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        if weight > 0.8:
            color = 'rgba(233, 69, 96, 0.8)'
        elif weight > 0.6:
            color = 'rgba(255, 107, 107, 0.6)'
        elif weight > 0.4:
            color = 'rgba(255, 217, 61, 0.5)'
        else:
            color = 'rgba(150, 150, 150, 0.3)'

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight*5, color=color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    sector_colors = {
        'Tech/Auto': '#00d9ff',
        'Battery': '#e94560',
        'Lithium': '#ff9f00',
        'Copper': '#00ff9f',
        'Recycling': '#9b59b6'
    }

    node_traces = []
    for sector, color in sector_colors.items():
        nodes_in_sector = [n for n in G.nodes() if G.nodes[n]['sector'] == sector]
        if not nodes_in_sector:
            continue

        node_x = [pos[n][0] for n in nodes_in_sector]
        node_y = [pos[n][1] for n in nodes_in_sector]

        sizes = []
        for node in nodes_in_sector:
            if node == 'TSLA':
                sizes.append(80)
            elif node in ring1:
                sizes.append(50)
            elif node in ring2:
                sizes.append(35)
            else:
                sizes.append(25)

        hover_text = []
        for node in nodes_in_sector:
            degree = G.degree(node)
            hover_text.append(f"<b>{node}</b><br>Sector: {sector}<br>Connections: {degree}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=nodes_in_sector,
            textposition='top center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=sizes,
                color=color,
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            name=sector,
            showlegend=True
        )
        node_traces.append(node_trace)

    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=dict(
            text='Market Attention Flow: Network Topology<br><sub>Zero-Shot Attention Propagation Model</sub>',
            font=dict(size=24, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='#533483',
            borderwidth=2,
            font=dict(size=12, color='white')
        ),
        hovermode='closest',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1920,
        height=1080,
        font=dict(color='white')
    )

    fig.add_annotation(
        text='Graph Neural Network | Unsupervised Attention Detection',
        xref='paper', yref='paper',
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=11, color='#666'),
        xanchor='right'
    )

    fig.write_html('results/hs4_interactive.html')

    fig_mpl, ax_mpl = plt.subplots(figsize=(19.2, 10.8), facecolor='#0a0a0a')
    ax_mpl.set_facecolor('#0a0a0a')

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        if weight > 0.8:
            color = '#e94560'
        elif weight > 0.6:
            color = '#ff6b6b'
        elif weight > 0.4:
            color = '#ffd93d'
        else:
            color = '#999999'

        ax_mpl.plot([x0, x1], [y0, y1], linewidth=weight*5, color=color, alpha=0.7, zorder=1)

    sector_colors_mpl = {
        'Tech/Auto': '#00d9ff',
        'Battery': '#e94560',
        'Lithium': '#ff9f00',
        'Copper': '#00ff9f',
        'Recycling': '#9b59b6'
    }

    for node in G.nodes():
        x, y = pos[node]
        sector = G.nodes[node]['sector']
        color = sector_colors_mpl.get(sector, '#666666')

        if node == 'TSLA':
            size = 1500
        elif node in ring1:
            size = 900
        elif node in ring2:
            size = 600
        else:
            size = 400

        ax_mpl.scatter(x, y, s=size, c=color, edgecolors='white', linewidths=3, zorder=2)
        ax_mpl.text(x, y+0.35, node, ha='center', va='bottom', fontsize=12,
                   weight='bold', color='white', zorder=3,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=color, alpha=0.9))

    ax_mpl.set_xlim(-6, 6)
    ax_mpl.set_ylim(-6, 6)
    ax_mpl.set_aspect('equal')
    ax_mpl.axis('off')
    ax_mpl.set_title('Market Attention Flow: Network Topology\nZero-Shot Attention Propagation Model',
                    fontsize=24, weight='bold', color='white', pad=30)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='white', label=sector)
                      for sector, color in sector_colors_mpl.items()]
    ax_mpl.legend(handles=legend_elements, loc='upper left', fontsize=12,
                 framealpha=0.9, facecolor='#1a1a2e', edgecolor='white')

    plt.tight_layout()
    plt.savefig('results/hs4.png', dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()

    print("  ✓ Saved to results/hs4.png and hs4_interactive.html")

def create_hs5_decay_curve():
    
    print("Creating HS5: Attention Decay - Action Window Analysis...")

    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    ax.set_facecolor('white')

    t = np.linspace(0, 6, 500)
    intensity = np.exp(-0.3 * t)

    ax.axvspan(0, 0.5, alpha=0.15, color='#ff4444', label='Peak Attention', zorder=0)
    ax.axvspan(0.5, 2, alpha=0.15, color='#ff9f00', label='Actionable Window', zorder=0)
    ax.axvspan(2, 6, alpha=0.15, color='#999999', label='Decayed Signal', zorder=0)

    ax.plot(t, intensity, linewidth=4, color='#1a1a2e', label='Attention Intensity: I(t) = e^(-0.3t)', zorder=5)

    half_life = np.log(2) / 0.3
    ax.axvline(x=half_life, color='#e94560', linestyle='--', linewidth=2.5, alpha=0.8, zorder=3)

    ax.annotate('Peak Attention\nNo edge yet',
                xy=(0.25, 0.85), fontsize=13, weight='bold', color='#ff4444',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#ff4444', linewidth=2))

    ax.annotate('ACTIONABLE WINDOW\nEdge lives here',
                xy=(1.25, 0.55), fontsize=14, weight='bold', color='#ff9f00',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#ff9f00', linewidth=3))

    ax.annotate('Too Late\nSignal decayed',
                xy=(4, 0.25), fontsize=13, weight='bold', color='#666666',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#999999', linewidth=2))

    ax.annotate(f'Half-life:\n~{half_life:.1f} hours',
                xy=(half_life, 0.5), xytext=(half_life + 0.8, 0.4),
                fontsize=12, weight='bold', color='#e94560',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#e94560', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#e94560', lw=2.5))

    ax.set_xlabel('Time (hours)', fontsize=16, weight='bold', color='#1a1a2e')
    ax.set_ylabel('Attention Intensity', fontsize=16, weight='bold', color='#1a1a2e')
    ax.set_title('Attention Decay: The Actionable Window\nNews doesn\'t fade linearly—it dies exponentially',
                 fontsize=20, weight='bold', color='#1a1a2e', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#cccccc')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='#1a1a2e', fancybox=True)

    plt.tight_layout()
    plt.savefig('results/hs5.png', dpi=300, facecolor='white', bbox_inches='tight')
    print("  ✓ Saved to results/hs5.png")
    plt.close()

def create_hs6_bridge_nodes():
    
    print("Creating HS6: Bridge Node Analysis - Network Position Matters...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8), facecolor='#f5f5f7')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#f5f5f7')

    G_tesla = nx.Graph()

    sectors_tesla = {
        'Auto': ['GM', 'F', 'TM'],
        'Battery': ['CATL', 'PANW', 'LG'],
        'Energy': ['ENPH', 'SEDG'],
        'Materials': ['ALB', 'SQM', 'LAC'],
        'Tech': ['NVDA', 'AMD']
    }

    pos_tesla = {'TSLA': (0, 0)}
    G_tesla.add_node('TSLA')

    angle_step = 2 * np.pi / len(sectors_tesla)
    for i, (sector, companies) in enumerate(sectors_tesla.items()):
        angle = i * angle_step
        sector_center_x = 2 * np.cos(angle)
        sector_center_y = 2 * np.sin(angle)

        for j, company in enumerate(companies):
            sub_angle = angle + (j - 1) * 0.3
            x = sector_center_x + 0.8 * np.cos(sub_angle)
            y = sector_center_y + 0.8 * np.sin(sub_angle)
            pos_tesla[company] = (x, y)
            G_tesla.add_node(company, sector=sector)
            G_tesla.add_edge('TSLA', company, weight=0.7 + np.random.uniform(0, 0.3))

    sector_colors_map = {
        'Auto': '#e94560', 'Battery': '#ff9f00', 'Energy': '#00ff9f',
        'Materials': '#ffd93d', 'Tech': '#00d9ff'
    }

    for u, v, data in G_tesla.edges(data=True):
        weight = data.get('weight', 0.5)
        ax1.plot([pos_tesla[u][0], pos_tesla[v][0]],
                [pos_tesla[u][1], pos_tesla[v][1]],
                linewidth=weight*4, color='#ff9f00', alpha=0.7, zorder=1)

    for node in G_tesla.nodes():
        x, y = pos_tesla[node]
        if node == 'TSLA':
            circle_outer = Circle((x, y), 0.25, color='#ff4444', alpha=0.3, zorder=3)
            circle_mid = Circle((x, y), 0.18, color='#ff4444', alpha=0.6, zorder=4)
            circle_inner = Circle((x, y), 0.12, color='#ffffff', zorder=5)
            ax1.add_patch(circle_outer)
            ax1.add_patch(circle_mid)
            ax1.add_patch(circle_inner)
            ax1.text(x, y, 'TSLA', ha='center', va='center', fontsize=14,
                    weight='bold', color='#1a1a2e', zorder=6)
        else:
            sector = G_tesla.nodes[node].get('sector', 'Auto')
            color = sector_colors_map.get(sector, '#666666')
            circle = Circle((x, y), 0.15, color=color, ec='black', linewidth=2, zorder=2)
            ax1.add_patch(circle)
            ax1.text(x, y-0.35, node, ha='center', va='top', fontsize=9,
                    weight='bold', color='black', zorder=2)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('TESLA: Bridge Node\nConnects 5 sectors • 15 connections',
                 fontsize=18, weight='bold', color='black', pad=20)

    G_ford = nx.Graph()

    auto_companies = ['GM', 'TM', 'HMC', 'STLA', 'NSANY']
    pos_ford = {'F': (0, 0)}
    G_ford.add_node('F')

    angles_ford = np.linspace(0, 2*np.pi, len(auto_companies), endpoint=False)
    for i, company in enumerate(auto_companies):
        x = 2.5 * np.cos(angles_ford[i])
        y = 2.5 * np.sin(angles_ford[i])
        pos_ford[company] = (x, y)
        G_ford.add_node(company)
        if i < 3:
            G_ford.add_edge('F', company, weight=0.4 + np.random.uniform(0, 0.2))

    for u, v, data in G_ford.edges(data=True):
        weight = data.get('weight', 0.4)
        ax2.plot([pos_ford[u][0], pos_ford[v][0]],
                [pos_ford[u][1], pos_ford[v][1]],
                linewidth=weight*3, color='#666666', alpha=0.5, zorder=1)

    for node in G_ford.nodes():
        x, y = pos_ford[node]
        if node == 'F':
            circle = Circle((x, y), 0.15, color='#999999', ec='black', linewidth=2, zorder=3)
            ax2.add_patch(circle)
            ax2.text(x, y, 'F', ha='center', va='center', fontsize=14,
                    weight='bold', color='white', zorder=4)
        else:
            circle = Circle((x, y), 0.12, color='#444444', ec='black', linewidth=1.5, zorder=2)
            ax2.add_patch(circle)
            ax2.text(x, y-0.3, node, ha='center', va='top', fontsize=9,
                    weight='bold', color='black', zorder=2)

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('FORD: Isolated Node\nSingle sector • 3 connections',
                 fontsize=18, weight='bold', color='black', pad=20)

    fig.suptitle('Why Some Companies Are Information Highways\nNetwork position matters as much as fundamentals',
                 fontsize=22, weight='bold', color='black', y=0.96)

    fig.text(0.25, 0.05, 'Tesla bridges multiple sectors\n→ High attention propagation\n→ Information hub',
            ha='center', fontsize=13, color='#ff6600', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#ff6600', linewidth=2))

    fig.text(0.75, 0.05, 'Ford isolated in one sector\n→ Limited propagation\n→ Information silo',
            ha='center', fontsize=13, color='#666666', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#666666', linewidth=2))

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig('results/hs6.png', dpi=300, facecolor='#f5f5f7', bbox_inches='tight')
    print("  ✓ Saved to results/hs6.png")
    plt.close()

def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()

    create_hs4_neon_network()
    print()
    create_hs5_decay_curve()
    print()
    create_hs6_bridge_nodes()

    print()
    print("=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nGenerated:")
    print("  results/hs4.png - Network Topology")
    print("  results/hs4_interactive.html - Interactive version")
    print("  results/hs5.png - Decay Curve")
    print("  results/hs6.png - Bridge Node Comparison")
    print("=" * 60)

if __name__ == '__main__':
    main()
