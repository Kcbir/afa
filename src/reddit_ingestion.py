import praw
from textblob import TextBlob
from datetime import datetime, timedelta
import re
from typing import List, Dict
import time

CLIENT_ID = "CDmKLXHk4gxMn3LZZq4n_Q"
CLIENT_SECRET = None
USER_AGENT = "MarketAttentionBot/1.0"

COMPANY_TICKERS = {
    'TSLA': ['tesla', 'tsla', 'elon', 'cybertruck', 'model 3', 'model y'],
    'BYD': ['byd', 'build your dreams'],
    'NIO': ['nio'],
    'RIVN': ['rivian', 'rivn', 'r1t', 'r1s'],
    'LCID': ['lucid', 'lcid', 'lucid motors'],
    'F': ['ford', 'f150', 'mustang mach'],
    'GM': ['gm', 'general motors', 'chevy', 'bolt'],
    'CATL': ['catl', 'contemporary amperex'],
    'ALB': ['albemarle', 'alb', 'lithium'],
    'SQM': ['sqm', 'sociedad quimica'],
    'LAC': ['lithium americas', 'lac'],
    'QS': ['quantumscape', 'solid state battery'],
    'CHPT': ['chargepoint', 'chpt'],
    'BLNK': ['blink', 'blnk', 'blink charging'],
    'NVDA': ['nvidia', 'nvda'],
    'AMD': ['amd', 'ryzen'],
}

def get_reddit_readonly():
    
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret="",
        user_agent=USER_AGENT,
        check_for_async=False
    )

def extract_sentiment(text: str) -> float:
    
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0

def find_mentioned_companies(text: str) -> List[str]:
    
    text_lower = text.lower()
    mentioned = []

    for ticker, keywords in COMPANY_TICKERS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                mentioned.append(ticker)
                break

    return mentioned

def scrape_reddit_events(subreddits: List[str] = None,
                        hours_back: int = 24,
                        limit: int = 100) -> List[Dict]:
    
    if subreddits is None:
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'electricvehicles', 'teslamotors']

    print(f"\nConnecting to Reddit API (read-only mode)...")

    try:
        reddit = get_reddit_readonly()
        print(f"✓ Connected to Reddit")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nNote: Reddit may require authentication. Using synthetic data instead.")
        return []

    events = []
    cutoff_time = datetime.now() - timedelta(hours=hours_back)

    print(f"\nScraping {len(subreddits)} subreddits...")

    for subreddit_name in subreddits:
        try:
            print(f"  → r/{subreddit_name}")
            subreddit = reddit.subreddit(subreddit_name)

            for post in subreddit.hot(limit=limit):
                post_time = datetime.fromtimestamp(post.created_utc)

                if post_time < cutoff_time:
                    continue

                text = f"{post.title} {post.selftext}"

                companies = find_mentioned_companies(text)

                if not companies:
                    continue

                sentiment = extract_sentiment(text)

                text_lower = text.lower()
                event_type = 'general_mention'

                if any(word in text_lower for word in ['earnings', 'beat', 'miss', 'revenue']):
                    event_type = 'earnings_report'
                elif any(word in text_lower for word in ['partnership', 'deal', 'contract']):
                    event_type = 'partnership'
                elif any(word in text_lower for word in ['breakthrough', 'innovation', 'launch']):
                    event_type = 'product_launch'
                elif any(word in text_lower for word in ['recall', 'delay', 'problem']):
                    event_type = 'negative_news'
                elif any(word in text_lower for word in ['upgrade', 'target', 'bullish']):
                    event_type = 'analyst_upgrade'
                elif any(word in text_lower for word in ['downgrade', 'bearish', 'overvalued']):
                    event_type = 'analyst_downgrade'

                primary_company = companies[0]

                event = {
                    'company': primary_company,
                    'type': event_type,
                    'sentiment': sentiment,
                    'timestamp': post_time.isoformat(),
                    'source': f'r/{subreddit_name}',
                    'url': f"https://reddit.com{post.permalink}",
                    'title': post.title[:100],
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'mentioned_companies': companies,
                    'hours_ago': (datetime.now() - post_time).total_seconds() / 3600
                }

                events.append(event)

            time.sleep(0.5)

        except Exception as e:
            print(f"  ✗ Error scraping r/{subreddit_name}: {e}")
            continue

    events.sort(key=lambda x: x['hours_ago'])

    print(f"\n✓ Found {len(events)} events with company mentions")

    if events:
        print("\nTop events:")
        for event in events[:5]:
            print(f"  • {event['company']} - {event['type']} (sentiment: {event['sentiment']:.2f})")
            print(f"    '{event['title']}' - {event['score']} upvotes")

    return events

def reddit_events_to_synthetic_format(reddit_events: List[Dict],
                                      G) -> List[Dict]:
    
    from src.synthetic_data import generate_synthetic_events

    if not reddit_events:
        print("\nNo Reddit events found. Using synthetic data.")
        return generate_synthetic_events(30)

    print(f"\nConverting {len(reddit_events)} Reddit events to propagation format...")

    formatted_events = []
    start_time = datetime.now() - timedelta(hours=24)

    for i, event in enumerate(reddit_events[:30]):
        from src.propagation import propagate_attention

        primary_company = event['company']
        sentiment = event['sentiment']
        intensity = abs(sentiment)

        scores = propagate_attention(G, primary_company, intensity)

        outcomes = {}
        for ticker, score in scores.items():
            if score > 0.1:
                base_move = sentiment * score * 0.05
                import numpy as np
                noise = np.random.normal(0, 0.003)
                outcomes[ticker] = base_move + noise

        event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')) if isinstance(event['timestamp'], str) else event['timestamp']
        hours_from_start = (event_time - start_time).total_seconds() / 3600

        formatted_event = {
            'id': i,
            'company': primary_company,
            'type': event['type'],
            'sentiment': sentiment,
            'timestamp': event['timestamp'],
            'hours_from_start': hours_from_start,
            'outcomes': outcomes,
            'source_weight': min(1.0, event.get('score', 10) / 100),
            'reddit_title': event.get('title', ''),
            'reddit_url': event.get('url', ''),
            'reddit_score': event.get('score', 0)
        }

        formatted_events.append(formatted_event)

    print(f"✓ Converted {len(formatted_events)} events")
    return formatted_events

if __name__ == '__main__':
    events = scrape_reddit_events(hours_back=24, limit=50)

    if events:
        print(f"\n{'='*60}")
        print(f"Found {len(events)} Reddit events!")
        print(f"{'='*60}")

        from collections import Counter
        company_counts = Counter(e['company'] for e in events)
        print("\nMost mentioned companies:")
        for company, count in company_counts.most_common(10):
            avg_sentiment = sum(e['sentiment'] for e in events if e['company'] == company) / count
            print(f"  {company}: {count} mentions (avg sentiment: {avg_sentiment:.2f})")
    else:
        print("\nNo events found or Reddit API unavailable.")
