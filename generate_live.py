#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.graph_setup import build_market_graph
from src.reddit_ingestion import scrape_reddit_events, reddit_events_to_synthetic_format
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
from src.synthetic_data import create_time_series
import json

def main():
    print("=" * 60)
    print("MARKET ATTENTION CONTAGION - LIVE REDDIT DATA")
    print("=" * 60)

    print("\n[1/6] Building market network graph...")
    G = build_market_graph()
    print(f"  → Created graph with {G.number_of_nodes()} companies and {G.number_of_edges()} relationships")

    print("\n[2/6] Scraping Reddit for live market events...")
    print("  Subreddits: wallstreetbets, stocks, investing, electricvehicles, teslamotors")
    print("  Time window: Last 24 hours")

    reddit_events = scrape_reddit_events(
        subreddits=['wallstreetbets', 'stocks', 'investing', 'electricvehicles', 'teslamotors'],
        hours_back=24,
        limit=100
    )

    print("\n[3/6] Converting Reddit events to propagation format...")
    events = reddit_events_to_synthetic_format(reddit_events, G)

    if not events:
        print("\n⚠ No Reddit events found or API unavailable.")
        print("This could be because:")
        print("  1. Reddit API requires authentication (client_secret needed)")
        print("  2. Rate limits hit")
        print("  3. No relevant posts in last 24 hours")
        print("\nFalling back to demo with synthetic data...")
        print("Run: python generate_results.py")
        return

    os.makedirs('data', exist_ok=True)
    with open('data/live_events.json', 'w') as f:
        json.dump(events, f, indent=2)

    print(f"  → Converted {len(events)} Reddit events")
    print(f"  → Saved to data/live_events.json")

    from collections import Counter
    company_counts = Counter(e['company'] for e in events)
    print("\n  Most mentioned companies:")
    for company, count in company_counts.most_common(5):
        print(f"     • {company}: {count} events")

    os.makedirs('results', exist_ok=True)

    print("\n[4/6] Creating visualizations with LIVE data...")

    print("  → Animated network...")
    create_animated_network(G, events, 'results/live_animated_network.html')

    print("  → Contagion heatmap...")
    create_heatmap(G, 'results/live_contagion_heatmap.html')

    print("  → Timeline ripple...")
    create_timeline_ripple(G, events, 'results/live_timeline_ripple.html')

    print("  → Influence ranking...")
    create_influence_ranking(G, 'results/live_influence_ranking.html')

    print("\n[5/6] Running Granger causality analysis...")
    companies = list(G.nodes())
    time_series = create_time_series(events, companies)

    granger_results = granger_causality_analysis(time_series, max_lag=6)
    significant_count = sum(1 for v in granger_results.values() if v['significant'])

    print(f"  → Tested {len(granger_results)} pairs")
    print(f"  → Found {significant_count} significant causal relationships (p < 0.05)")

    if significant_count > 0:
        create_granger_network(granger_results, 'results/live_granger_network.html')

    print("\n[6/6] Running strategy backtest on LIVE data...")
    metrics = backtest_propagation_strategy(G, events, threshold=0.25)

    if 'error' not in metrics:
        visualize_backtest(metrics, 'results/live_backtest_results.html')
        print(f"  → {metrics['total_trades']} trades executed")
        print(f"  → Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  → Total Return: {metrics['total_return']*100:.2f}%")

    print("\n" + "=" * 60)
    print("✓ LIVE RESULTS GENERATED!")
    print("=" * 60)
    print("\nGenerated files (with LIVE Reddit data):")
    print("  • results/live_animated_network.html")
    print("  • results/live_contagion_heatmap.html")
    print("  • results/live_timeline_ripple.html")
    print("  • results/live_influence_ranking.html")
    print("  • results/live_granger_network.html")
    print("  • results/live_backtest_results.html")
    print("  • data/live_events.json")

    print("\n🔥 Open results/live_animated_network.html to see REAL market attention!")
    print("=" * 60)

    if events:
        print("\n📊 LIVE DATA STATS:")
        print(f"  Total events: {len(events)}")
        print(f"  Time span: {events[-1]['hours_from_start']:.1f} hours")
        print(f"  Avg sentiment: {sum(e['sentiment'] for e in events)/len(events):.2f}")

        bullish = sum(1 for e in events if e['sentiment'] > 0.2)
        bearish = sum(1 for e in events if e['sentiment'] < -0.2)
        print(f"  Bullish events: {bullish}")
        print(f"  Bearish events: {bearish}")

        if 'reddit_score' in events[0]:
            top_event = max(events, key=lambda x: x.get('reddit_score', 0))
            print(f"\n  🔥 Hottest Reddit post:")
            print(f"     {top_event.get('reddit_title', 'N/A')}")
            print(f"     {top_event['company']} - {top_event['reddit_score']} upvotes")
            print(f"     Sentiment: {top_event['sentiment']:.2f}")

if __name__ == '__main__':
    main()
