#!/usr/bin/env python3

import webbrowser
import os
import sys

RESULTS = {
    'network': 'results/animated_network.html',
    'heatmap': 'results/contagion_heatmap.html',
    'timeline': 'results/timeline_ripple.html',
    'granger': 'results/granger_network.html',
    'influence': 'results/influence_ranking.html',
    'backtest': 'results/backtest_results.html',
}

def main():
    if len(sys.argv) < 2:
        print("Market Attention Contagion - Visualization Viewer")
        print("\nAvailable visualizations:")
        print("  network   - Animated propagation network (MAIN DEMO)")
        print("  heatmap   - Contagion strength matrix")
        print("  timeline  - Temporal ripple visualization")
        print("  granger   - Statistical causality network")
        print("  influence - Top influencer ranking")
        print("  backtest  - Strategy performance metrics")
        print("  all       - Open all visualizations")
        print("\nUsage: python view_results.py [name]")
        print("Example: python view_results.py network")
        return

    choice = sys.argv[1].lower()

    if choice == 'all':
        print("Opening all visualizations...")
        for name, path in RESULTS.items():
            if os.path.exists(path):
                print(f"  → {name}")
                webbrowser.open('file://' + os.path.abspath(path))
        print("\nAll visualizations opened in browser!")
    elif choice in RESULTS:
        path = RESULTS[choice]
        if os.path.exists(path):
            print(f"Opening {choice} visualization...")
            webbrowser.open('file://' + os.path.abspath(path))
            print(f"✓ Opened {path}")
        else:
            print(f"Error: {path} not found. Run generate_results.py first.")
    else:
        print(f"Unknown visualization: {choice}")
        print("Run without arguments to see available options.")

if __name__ == '__main__':
    main()
