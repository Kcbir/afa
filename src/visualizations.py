import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

def create_network_layout(G: nx.DiGraph) -> Dict[str, tuple]:
    
    sectors = {}
    for node, data in G.nodes(data=True):
        sector = data.get('sector', 'other')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(node)

    pos = {}
    sector_angles = np.linspace(0, 2*np.pi, len(sectors), endpoint=False)

    for (sector, nodes), angle in zip(sectors.items(), sector_angles):
        sector_center = (3 * np.cos(angle), 3 * np.sin(angle))
        n_nodes = len(nodes)

        for i, node in enumerate(nodes):
            sub_angle = 2 * np.pi * i / n_nodes
            radius = 0.8 + np.random.uniform(-0.1, 0.1)
            x = sector_center[0] + radius * np.cos(sub_angle)
            y = sector_center[1] + radius * np.sin(sub_angle)
            pos[node] = (x, y)

    return pos

def create_animated_network(G: nx.DiGraph,
                           events: List[Dict],
                           output_file: str = 'results/animated_network.html'):
    
    from src.propagation import multi_source_propagation

    pos = create_network_layout(G)

    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )

    frames = []
    max_time = max(e['hours_from_start'] for e in events) + 12

    for t in np.arange(0, max_time, 0.5):
        active_events = [
            (e['company'], abs(e['sentiment']), e['hours_from_start'])
            for e in events if e['hours_from_start'] <= t
        ]

        scores = multi_source_propagation(G, active_events, t, lambda_decay=0.3)

        max_score = max(scores.values()) if scores else 1.0
        normalized_scores = {k: v/max_score for k, v in scores.items()}

        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            score = normalized_scores.get(node, 0)
            node_colors.append(score)

            size = G.nodes[node].get('size', 30)
            node_sizes.append(size * (0.5 + score))

            node_text.append(f"{node}<br>Attention: {score:.3f}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='YlOrRd',
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Attention",
                    thickness=15,
                    len=0.7
                ),
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )

        frame = go.Frame(
            data=[*edge_trace, node_trace],
            name=f'{t:.1f}',
            layout=go.Layout(
                title_text=f'Market Attention Propagation - Hour {t:.1f}'
            )
        )
        frames.append(frame)

    fig = go.Figure(
        data=[*edge_trace, frames[0].data[-1]],
        frames=frames
    )

    fig.update_layout(
        title='Market Attention Contagion Network',
        showlegend=False,
        hovermode='closest',
        width=1400,
        height=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='white', size=12),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.1
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': f.name,
                    'method': 'animate'
                }
                for f in frames
            ],
            'x': 0.1,
            'len': 0.9,
            'y': 0
        }]
    )

    fig.write_html(output_file)
    print(f"Saved animated network to {output_file}")

def create_heatmap(G: nx.DiGraph, output_file: str = 'results/contagion_heatmap.html'):
    
    from src.propagation import propagate_attention

    companies = list(G.nodes())
    n = len(companies)
    matrix = np.zeros((n, n))

    for i, source in enumerate(companies):
        scores = propagate_attention(G, source, 1.0)
        for j, target in enumerate(companies):
            matrix[i][j] = scores.get(target, 0)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=companies,
        y=companies,
        colorscale='YlOrRd',
        text=matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Propagation<br>Strength")
    ))

    fig.update_layout(
        title='Market Attention Contagion Matrix',
        xaxis_title='Target Company',
        yaxis_title='Source Company',
        width=1200,
        height=1100,
        font=dict(size=10),
        xaxis=dict(tickangle=-45)
    )

    fig.write_html(output_file)
    print(f"Saved heatmap to {output_file}")

    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix,
                xticklabels=companies,
                yticklabels=companies,
                cmap='YlOrRd',
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Propagation Strength'},
                linewidths=0.1)
    plt.title('Attention Contagion Matrix', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Target Company', fontsize=12)
    plt.ylabel('Source Company', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file.replace('.html', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_timeline_ripple(G: nx.DiGraph,
                          events: List[Dict],
                          output_file: str = 'results/timeline_ripple.html'):
    
    from src.propagation import propagate_with_time

    companies = list(G.nodes())[:20]
    max_time = 72

    time_points = np.arange(0, max_time, 0.5)
    attention_data = {company: [] for company in companies}

    for t in time_points:
        active_events = [
            (e['company'], abs(e['sentiment']), e['hours_from_start'])
            for e in events if e['hours_from_start'] <= t
        ]

        from src.propagation import multi_source_propagation
        scores = multi_source_propagation(G, active_events, t)

        for company in companies:
            attention_data[company].append(scores.get(company, 0))

    fig = go.Figure()

    for i, company in enumerate(companies):
        fig.add_trace(go.Scatter(
            x=time_points,
            y=[i] * len(time_points),
            mode='markers',
            marker=dict(
                size=8,
                color=attention_data[company],
                colorscale='YlOrRd',
                cmin=0,
                cmax=max(max(attention_data[c]) for c in companies),
                line=dict(width=0.5, color='white')
            ),
            name=company,
            hovertemplate=f'{company}<br>Time: %{{x:.1f}}h<br>Attention: %{{marker.color:.3f}}<extra></extra>'
        ))

    for event in events:
        if event['hours_from_start'] < max_time:
            company_idx = companies.index(event['company']) if event['company'] in companies else None
            if company_idx is not None:
                fig.add_vline(
                    x=event['hours_from_start'],
                    line_dash="dash",
                    line_color="cyan",
                    opacity=0.3,
                    annotation_text=event['type'][:10],
                    annotation_position="top"
                )

    fig.update_layout(
        title='Attention Ripple Timeline',
        xaxis_title='Time (hours)',
        yaxis=dict(
            ticktext=companies,
            tickvals=list(range(len(companies))),
            title='Company'
        ),
        width=1400,
        height=800,
        hovermode='closest',
        showlegend=False
    )

    fig.write_html(output_file)
    print(f"Saved timeline to {output_file}")

def create_influence_ranking(G: nx.DiGraph, output_file: str = 'results/influence_ranking.html'):
    
    from src.propagation import find_critical_nodes

    critical_nodes = find_critical_nodes(G, top_n=15)

    tickers = [node for node, _ in critical_nodes]
    scores = [score for _, score in critical_nodes]

    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=tickers,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='YlOrRd',
                colorbar=dict(title="Total Influence")
            ),
            text=[f'{s:.2f}' for s in scores],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Network Influence Leaders',
        xaxis_title='Total Propagation Influence',
        yaxis_title='Company',
        width=900,
        height=600,
        yaxis=dict(autorange='reversed')
    )

    fig.write_html(output_file)
    print(f"Saved influence ranking to {output_file}")
