# 🚀 Quick Start - Everything You Need

## For Blog Post RIGHT NOW

Just need the 3 main visuals? Run this:

```bash
python create_visualizations_v2.py
```

This generates:
- `results/hs4.png` - Network topology
- `results/hs5.png` - Decay curve (white background, cleanest)
- `results/hs6.png` - Bridge node comparison

**Time:** 10 seconds  
**Use:** All 3 in your blog post

---

## For Full Demo (Interactive + Data)

Want everything including the animated visualizations?

```bash
python demo.py
```

This runs the full demo and opens the animated network in your browser.

**Time:** 30 seconds  
**Opens:** Animated network visualization automatically

---

## For Live Reddit Data (Experimental)

Want to pull real data from Reddit?

**First:** Add your Reddit credentials to `src/reddit_ingestion.py`
- Your client_id: `CDmKLXHk4gxMn3LZZq4n_Q`
- Need: `client_secret` (get from reddit.com/prefs/apps)

**Then:**
```bash
python generate_live.py
```

This scrapes r/wallstreetbets and other subs for real market events.

**Time:** 60 seconds  
**Result:** Live visualizations in `results/live_*.html`

---

## Cheat Sheet

| Command | What | Time | Output |
|---------|------|------|--------|
| `python create_visualizations_v2.py` | 3 blog visuals | 10s | hs4, hs5, hs6 |
| `python demo.py` | Full demo + auto-open | 30s | All HTML + opens browser |
| `python generate_results.py` | All analysis | 30s | All results + data |
| `python generate_live.py` | Real Reddit data | 60s | Live results |
| `python view_results.py network` | View specific viz | 1s | Opens in browser |

---

## Files You Care About

**For blog post:**
- `results/hs4.png` - Main network graph
- `results/hs5.png` - The decay curve (explains the edge)
- `results/hs6.png` - Why Tesla (bridge nodes)

**For interactive demo:**
- `results/hs4_interactive.html` - Network (interactive)
- `results/animated_network.html` - Main demo (auto-plays)

**For deep dive:**
- `results/backtest_results.html` - Performance metrics
- `results/granger_network.html` - Statistical causality
- `RESULTS_SUMMARY.md` - Full analysis writeup

---

## One-Liner for Blog

```bash
python create_visualizations_v2.py && open results/hs5.png
```

This makes the 3 visuals and opens the cleanest one (decay curve).

---

## Troubleshooting

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"No events found" (Live Reddit)**
- Need client_secret in `src/reddit_ingestion.py`
- Or just use synthetic data (works perfectly)

**"Can't open HTML"**
- They're in `results/` folder
- Double-click any `.html` file
- Or: `open results/animated_network.html`

---

## What You Have

✅ Network analysis (30 companies, 51 relationships)  
✅ Propagation math (decay, Markov chains, real formulas)  
✅ Synthetic events (30 realistic market events)  
✅ 7 interactive visualizations  
✅ 3 blog-ready static images  
✅ Granger causality analysis  
✅ Backtest with realistic metrics  
✅ Reddit integration (optional)  

**Total project size:** ~40MB  
**Time to demo:** 30 seconds  
**Time to understand:** 5 minutes looking at HS5  

---

## For Your Blog

1. Run: `python create_visualizations_v2.py`
2. Use: hs4, hs5, hs6
3. Write around the decay curve (hs5)
4. Show network (hs4) 
5. Explain bridge nodes (hs6)
6. Done.

The narrative writes itself.
