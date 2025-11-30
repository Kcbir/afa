import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

def granger_causality_analysis(time_series: Dict[str, np.ndarray],
                                max_lag: int = 6) -> Dict[Tuple[str, str], Dict]:
    
    results = {}
    companies = list(time_series.keys())

    for source in companies[:15]:
        for target in companies[:15]:
            if source == target:
                continue

            source_data = time_series[source]
            target_data = time_series[target]

            if len(source_data) < 2*max_lag or len(target_data) < 2*max_lag:
                continue

            data = pd.DataFrame({
                'target': target_data,
                'source': source_data
            })

            try:
                test_result = grangercausalitytests(data[['target', 'source']],
                                                    maxlag=max_lag,
                                                    verbose=False)

                p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                f_stats = [test_result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag+1)]

                min_p_idx = np.argmin(p_values)
                min_p_value = p_values[min_p_idx]
                best_lag = min_p_idx + 1

                results[(source, target)] = {
                    'p_value': min_p_value,
                    'f_stat': f_stats[min_p_idx],
                    'best_lag': best_lag,
                    'significant': min_p_value < 0.05
                }

            except Exception as e:
                continue

    return results

def create_granger_network(granger_results: Dict[Tuple[str, str], Dict],
                          output_file: str = 'results/granger_network.html'):
    
    significant = {k: v for k, v in granger_results.items() if v['significant']}

    if not significant:
        print("No significant Granger causality found")
        return

    G = nx.DiGraph()
    for (source, target), result in significant.items():
        G.add_edge(source, target, weight=result['f_stat'], p_value=result['p_value'])

    pos = nx.spring_layout(G, k=2, iterations=50)

    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_data = G[edge[0]][edge[1]]
        f_stat = edge_data['weight']
        p_value = edge_data['p_value']

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=2 + f_stat/10,
                color=f'rgba(255, {int(100 + p_value*1000)}, 100, 0.6)'
            ),
            hoverinfo='text',
            text=f'{edge[0]} → {edge[1]}<br>F-stat: {f_stat:.2f}<br>p-value: {p_value:.4f}',
            showlegend=False
        ))

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        degree = G.in_degree(node) + G.out_degree(node)
        node_sizes.append(20 + degree * 10)

        node_text.append(f'{node}<br>In: {G.in_degree(node)}, Out: {G.out_degree(node)}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color='#FF6B6B',
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )

    fig = go.Figure(data=[*edge_traces, node_trace])

    fig.update_layout(
        title='Granger Causality Network (p < 0.05)',
        showlegend=False,
        hovermode='closest',
        width=1200,
        height=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.write_html(output_file)
    print(f"Saved Granger network to {output_file}")

    return significant

def backtest_propagation_strategy(G: nx.DiGraph,
                                  events: List[Dict],
                                  threshold: float = 0.3) -> Dict:
    
    from src.propagation import propagate_attention

    trades = []
    positions = []

    for event in events:
        source = event['company']
        sentiment = event['sentiment']
        intensity = abs(sentiment)

        scores = propagate_attention(G, source, intensity)

        targets = sorted([(k, v) for k, v in scores.items() if k != source],
                        key=lambda x: x[1], reverse=True)

        for target, score in targets[:5]:
            if score < threshold:
                continue

            actual_move = event['outcomes'].get(target, 0)

            predicted_direction = 1 if sentiment > 0 else -1
            actual_direction = 1 if actual_move > 0 else -1

            correct = (predicted_direction == actual_direction and abs(actual_move) > 0.01)

            trade = {
                'event_id': event['id'],
                'source': source,
                'target': target,
                'predicted_score': score,
                'predicted_direction': predicted_direction,
                'actual_move': actual_move,
                'correct': correct,
                'return': actual_move * predicted_direction if abs(actual_move) > 0.01 else 0
            }

            trades.append(trade)

            positions.append({
                'time': event['hours_from_start'],
                'return': trade['return']
            })

    df = pd.DataFrame(trades)

    if len(df) == 0:
        return {'error': 'No trades generated'}

    accuracy = df['correct'].mean()
    total_return = df['return'].sum()
    mean_return = df['return'].mean()
    std_return = df['return'].std()
    sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

    cumulative_returns = np.cumsum([p['return'] for p in positions])

    metrics = {
        'total_trades': len(trades),
        'accuracy': accuracy,
        'total_return': total_return,
        'mean_return': mean_return,
        'sharpe_ratio': sharpe,
        'win_rate': (df['return'] > 0).mean(),
        'avg_win': df[df['return'] > 0]['return'].mean() if (df['return'] > 0).any() else 0,
        'avg_loss': df[df['return'] < 0]['return'].mean() if (df['return'] < 0).any() else 0,
        'cumulative_returns': cumulative_returns,
        'trades': trades
    }

    return metrics

def visualize_backtest(metrics: Dict, output_file: str = 'results/backtest_results.html'):
    
    if 'error' in metrics:
        print(f"Error in backtest: {metrics['error']}")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns',
            'Trade Distribution',
            'Performance Metrics',
            'Win/Loss Analysis'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )

    cum_returns = metrics['cumulative_returns']
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cum_returns))),
            y=cum_returns * 100,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )

    returns = [t['return'] for t in metrics['trades']]
    fig.add_trace(
        go.Histogram(
            x=np.array(returns) * 100,
            nbinsx=30,
            name='Returns',
            marker_color='#4169E1'
        ),
        row=1, col=2
    )

    metric_names = ['Accuracy', 'Win Rate', 'Sharpe Ratio']
    metric_values = [
        metrics['accuracy'] * 100,
        metrics['win_rate'] * 100,
        metrics['sharpe_ratio'] * 10
    ]

    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['#FF6B6B', '#4ECDC4', '#FFD93D'],
            text=[f'{v:.1f}' for v in metric_values],
            textposition='auto'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=['Avg Win', 'Avg Loss'],
            y=[metrics['avg_win'] * 100, metrics['avg_loss'] * 100],
            marker_color=['#00ff00', '#ff0000'],
            text=[f"{metrics['avg_win']*100:.2f}%", f"{metrics['avg_loss']*100:.2f}%"],
            textposition='auto'
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        title_text=f"Backtest Results - {metrics['total_trades']} trades, "
                   f"{metrics['accuracy']*100:.1f}% accuracy, "
                   f"{metrics['total_return']*100:.2f}% return",
        showlegend=False,
        height=900,
        width=1400
    )

    fig.write_html(output_file)
    print(f"Saved backtest visualization to {output_file}")

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Avg Win: {metrics['avg_win']*100:.2f}%")
    print(f"Avg Loss: {metrics['avg_loss']*100:.2f}%")
