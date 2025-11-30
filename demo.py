#!/usr/bin/env python3

import subprocess
import webbrowser
import os
import time

def main():
    print("=" * 60)
    print("MARKET ATTENTION CONTAGION - ONE-CLICK DEMO")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Generate all visualizations")
    print("  2. Open the main demo in your browser")
    print("  3. Show you the results")
    print("\nGenerating... (takes ~30 seconds)")
    print("=" * 60)

    result = subprocess.run(['python', 'generate_results.py'],
                          capture_output=False,
                          text=True)

    if result.returncode != 0:
        print("\n❌ Error generating results")
        return

    time.sleep(1)

    main_viz = os.path.abspath('results/animated_network.html')

    if os.path.exists(main_viz):
        print("\n" + "=" * 60)
        print("✓ SUCCESS!")
        print("=" * 60)
        print("\nOpening animated network visualization...")
        print("(If it doesn't open, go to: results/animated_network.html)")

        webbrowser.open('file://' + main_viz)

        print("\n📊 All visualizations available:")
        print("  • results/animated_network.html - MAIN DEMO (opening now)")
        print("  • results/contagion_heatmap.html - Propagation matrix")
        print("  • results/timeline_ripple.html - Temporal view")
        print("  • results/granger_network.html - Causal network")
        print("  • results/influence_ranking.html - Top influencers")
        print("  • results/backtest_results.html - Strategy performance")

        print("\n💡 TIP: Click the PLAY button to watch attention propagate!")
        print("=" * 60)
    else:
        print("\n❌ Results not found. Check for errors above.")

if __name__ == '__main__':
    main()
