# Market Attention Contagion - Quick Start

## Option 1: Demo with Synthetic Data (READY NOW)

This works immediately - just run it:

```bash
python generate_results.py
```

Then open the visualizations:
```bash
python view_results.py network
```

**What you get:**
- 30 realistic synthetic events
- Full propagation visualization
- Granger causality analysis
- Backtest with 79% accuracy
- All 7 visualizations

**Use this for:**
- Blog demos
- Proof of concept
- Quick visualization
- Testing the system

---

## Option 2: Live Reddit Data (Requires Setup)

To pull REAL data from Reddit, you need Reddit API credentials.

### Get Reddit API Credentials

1. Go to: https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **Name**: MarketAttentionBot
   - **App type**: Select "script"
   - **Description**: Market attention analysis
   - **About URL**: (leave blank)
   - **Redirect URI**: http://localhost:8080
4. Click "Create app"
5. You'll see:
   - **client_id**: Under "personal use script" (14 characters)
   - **client_secret**: The longer secret key (27 characters)

### Add Your Credentials

Edit [src/reddit_ingestion.py](src/reddit_ingestion.py#L11-L12):

```python
CLIENT_ID = "YOUR_CLIENT_ID_HERE"          # 14 chars
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"  # 27 chars
```

### Run with Live Data

```bash
python generate_live.py
```

This will:
- Scrape r/wallstreetbets, r/stocks, r/investing, r/electricvehicles, r/teslamotors
- Extract mentions of 15+ companies (TSLA, BYD, NIO, ALB, etc.)
- Analyze sentiment using TextBlob
- Generate visualizations with REAL market attention
- Save results to `results/live_*.html`

### View Live Results

```bash
# Open live animated network
open results/live_animated_network.html

# Or use the viewer
python view_results.py network
```

---

## What's the Difference?

| Feature | Synthetic Data | Live Reddit Data |
|---------|---------------|------------------|
| **Speed** | Instant | ~30-60 seconds |
| **Setup** | None | Reddit API key |
| **Data** | Pre-generated realistic events | Real r/wallstreetbets posts |
| **Accuracy** | 79% (synthetic) | Varies by Reddit activity |
| **Events** | Always 30 perfect events | Depends on last 24h of Reddit |
| **Use Case** | Demo, blog, proof-of-concept | Live monitoring, real analysis |

---

## Current Status

Right now you have:

✅ **Synthetic demo** - Fully working
- Run: `python generate_results.py`
- Files: `results/*.html`
- Status: **READY TO SHOW**

⚠️ **Live Reddit** - Needs authentication
- Run: `python generate_live.py`
- Status: Falls back to synthetic without credentials
- Fix: Add client_secret to [src/reddit_ingestion.py](src/reddit_ingestion.py)

---

## Quick Commands

```bash
# Generate synthetic demo (works now)
python generate_results.py

# View main visualization
python view_results.py network

# View all visualizations
python view_results.py all

# Try live Reddit (needs credentials)
python generate_live.py

# Test Reddit connection
python -c "from src.reddit_ingestion import scrape_reddit_events; scrape_reddit_events(limit=10)"
```

---

## Files Generated

### Synthetic Data
- `results/animated_network.html` - Main demo
- `results/contagion_heatmap.html` - Propagation matrix
- `results/timeline_ripple.html` - Temporal view
- `results/granger_network.html` - Causal relationships
- `results/influence_ranking.html` - Top influencers
- `results/backtest_results.html` - Strategy performance
- `data/synthetic_events.json` - Event data

### Live Reddit Data (when configured)
- `results/live_animated_network.html`
- `results/live_contagion_heatmap.html`
- `results/live_timeline_ripple.html`
- `results/live_granger_network.html`
- `results/live_influence_ranking.html`
- `results/live_backtest_results.html`
- `data/live_events.json` - Real Reddit events

---

## Troubleshooting

**"No events found"**
- Reddit API not configured or rate limited
- Falls back to synthetic data automatically
- Add credentials to [src/reddit_ingestion.py](src/reddit_ingestion.py)

**"Import error"**
- Run: `pip install -r requirements.txt`

**Visualizations won't open**
- They're HTML files - just double-click them
- Or: `open results/animated_network.html` (Mac)
- Or: `start results/animated_network.html` (Windows)

**Want to customize**
- Add companies: Edit [src/graph_setup.py](src/graph_setup.py)
- Change propagation: Edit [src/propagation.py](src/propagation.py)
- More events: Change `n_events` in [src/synthetic_data.py](src/synthetic_data.py)

---

## Next Steps

**For Blog:**
1. Run synthetic demo
2. Screenshot the animated network
3. Embed HTML files directly
4. Use [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for writeup

**For Live Monitoring:**
1. Get Reddit API credentials
2. Add to [src/reddit_ingestion.py](src/reddit_ingestion.py)
3. Run `python generate_live.py`
4. Set up cron job to run every hour

**For Production:**
1. Add more data sources (Twitter, news RSS)
2. Historical validation
3. Real-time alerts
4. Risk management

---

## Show Time!

Ready to demo RIGHT NOW:

```bash
python generate_results.py && python view_results.py network
```

This generates everything and opens the main visualization.

**The animated network is the star** - watch attention propagate through the company network in real-time. Hit play and watch the magic.
