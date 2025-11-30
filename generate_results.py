#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.graph_setup import build_market_graph
from src.synthetic_data import generate_synthetic_events, create_time_series
from src.visualizations import (
    create_animated_network,
    create_heatmap,
    create_timeline_ripple,
    create_influence_ranking
)
from src.analysis import (
    granger_causality_analysis,
    create_granger_network,
    backtest_propagation_strategy,
    visualize_backtest
)
import json

def main():
    print("=" * 60)
    print("MARKET ATTENTION CONTAGION - Generating Demo Results")
    print("=" * 60)

    print("\n[1/7] Building market network graph...")
    G = build_market_graph()
    print(f"  → Created graph with {G.number_of_nodes()} companies and {G.number_of_edges()} relationships")

    print("\n[2/7] Generating synthetic market events...")
    events = generate_synthetic_events(30)

    os.makedirs('data', exist_ok=True)
    with open('data/synthetic_events.json', 'w') as f:
        json.dump(events, f, indent=2)

    print(f"  → Generated {len(events)} events over 72 hours")
    print(f"  → Sample events:")
    for e in events[:3]:
        print(f"     • {e['company']} - {e['type']} (sentiment: {e['sentiment']:.2f})")

    os.makedirs('results', exist_ok=True)

    print("\n[3/7] Creating animated network visualization...")
    create_animated_network(G, events, 'results/animated_network.html')
    print("  → Animated network showing propagation over time")

    print("\n[4/7] Creating contagion heatmap...")
    create_heatmap(G, 'results/contagion_heatmap.html')
    print("  → Heatmap showing propagation strength between all companies")

    print("\n[5/7] Creating timeline ripple visualization...")
    create_timeline_ripple(G, events, 'results/timeline_ripple.html')
    print("  → Timeline showing attention waves across companies")

    print("\n[6/7] Creating influence ranking...")
    create_influence_ranking(G, 'results/influence_ranking.html')
    print("  → Ranking of most influential nodes in network")

    print("\n[7/7] Running Granger causality analysis...")
    companies = list(G.nodes())
    time_series = create_time_series(events, companies)

    granger_results = granger_causality_analysis(time_series, max_lag=6)
    significant_count = sum(1 for v in granger_results.values() if v['significant'])

    print(f"  → Tested {len(granger_results)} pairs")
    print(f"  → Found {significant_count} significant causal relationships (p < 0.05)")

    if significant_count > 0:
        create_granger_network(granger_results, 'results/granger_network.html')
        print("  → Created Granger causality network")

        top_granger = sorted(
            [(k, v) for k, v in granger_results.items() if v['significant']],
            key=lambda x: x[1]['f_stat'],
            reverse=True
        )[:5]

        print("\n  Top Granger causality relationships:")
        for (source, target), result in top_granger:
            print(f"     • {source} → {target}: F={result['f_stat']:.2f}, p={result['p_value']:.4f}")

    print("\n[Bonus] Running strategy backtest...")
    metrics = backtest_propagation_strategy(G, events, threshold=0.25)

    if 'error' not in metrics:
        visualize_backtest(metrics, 'results/backtest_results.html')
        print(f"  → Backtested on {metrics['total_trades']} trades")
        print(f"  → Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  → Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  → Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    print("\n" + "=" * 60)
    print("✓ ALL RESULTS GENERATED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • results/animated_network.html - Main interactive demo")
    print("  • results/contagion_heatmap.html - Propagation matrix")
    print("  • results/contagion_heatmap.png - Static heatmap")
    print("  • results/timeline_ripple.html - Temporal visualization")
    print("  • results/influence_ranking.html - Top influencers")
    print("  • results/granger_network.html - Causal relationships")
    print("  • results/backtest_results.html - Strategy performance")
    print("  • data/synthetic_events.json - Event data")

    print("\n🚀 Open results/animated_network.html to see the demo!")
    print("=" * 60)

if __name__ == '__main__':
    main()
