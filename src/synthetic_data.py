import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json

def generate_synthetic_events(n_events: int = 30) -> List[Dict]:
    
    np.random.seed(42)

    events_templates = [
        {
            'company': 'TSLA',
            'type': 'price_cut',
            'sentiment': 0.3,
            'base_outcomes': {
                'TSLA': 0.025, 'BYD': 0.018, 'NIO': 0.015,
                'CATL': -0.012, 'ALB': -0.022, 'SQM': -0.018,
                'RIVN': 0.008, 'LCID': 0.01
            }
        },
        {
            'company': 'TSLA',
            'type': 'production_beat',
            'sentiment': 0.8,
            'base_outcomes': {
                'TSLA': 0.055, 'CATL': 0.032, 'ALB': 0.038,
                'PANW': 0.028, 'SQM': 0.035, 'LAC': 0.025,
                'NVDA': 0.015, 'BYD': -0.008
            }
        },
        {
            'company': 'TSLA',
            'type': 'delivery_miss',
            'sentiment': -0.6,
            'base_outcomes': {
                'TSLA': -0.048, 'CATL': -0.025, 'ALB': -0.032,
                'NIO': -0.018, 'RIVN': 0.012, 'F': 0.008,
                'CHPT': -0.015
            }
        },

        {
            'company': 'BYD',
            'type': 'new_partnership',
            'sentiment': 0.75,
            'base_outcomes': {
                'BYD': 0.042, 'CATL': 0.038, 'ALB': 0.035,
                'LTHM': 0.028, 'SQM': 0.032, 'TSLA': -0.012,
                'NIO': -0.008
            }
        },
        {
            'company': 'BYD',
            'type': 'sales_record',
            'sentiment': 0.85,
            'base_outcomes': {
                'BYD': 0.062, 'CATL': 0.045, 'ALB': 0.042,
                'SQM': 0.038, 'NIO': 0.015, 'TSLA': -0.015
            }
        },

        {
            'company': 'ALB',
            'type': 'supply_shortage',
            'sentiment': 0.7,
            'base_outcomes': {
                'ALB': 0.068, 'SQM': 0.055, 'LAC': 0.058,
                'LTHM': 0.052, 'PLL': 0.045, 'TSLA': -0.018,
                'CATL': 0.022
            }
        },
        {
            'company': 'CATL',
            'type': 'new_battery_tech',
            'sentiment': 0.9,
            'base_outcomes': {
                'CATL': 0.072, 'TSLA': 0.035, 'BYD': 0.028,
                'NIO': 0.025, 'F': 0.018, 'GM': 0.015,
                'ALB': 0.032, 'QS': -0.025
            }
        },
        {
            'company': 'QS',
            'type': 'breakthrough_claim',
            'sentiment': 0.95,
            'base_outcomes': {
                'QS': 0.125, 'ENVX': 0.045, 'TSLA': 0.022,
                'ALB': -0.015, 'CATL': -0.028
            }
        },

        {
            'company': 'NVDA',
            'type': 'auto_chip_launch',
            'sentiment': 0.8,
            'base_outcomes': {
                'NVDA': 0.048, 'TSLA': 0.028, 'NIO': 0.022,
                'AMD': -0.018, 'MRVL': 0.015
            }
        },

        {
            'company': 'F',
            'type': 'ev_investment',
            'sentiment': 0.65,
            'base_outcomes': {
                'F': 0.038, 'LG': 0.032, 'CATL': 0.025,
                'ALB': 0.028, 'CHPT': 0.022, 'TSLA': -0.008
            }
        },
        {
            'company': 'GM',
            'type': 'battery_plant_announcement',
            'sentiment': 0.7,
            'base_outcomes': {
                'GM': 0.045, 'LG': 0.052, 'ALB': 0.035,
                'LTHM': 0.028, 'F': 0.012
            }
        },

        {
            'company': 'CHPT',
            'type': 'expansion_news',
            'sentiment': 0.6,
            'base_outcomes': {
                'CHPT': 0.055, 'BLNK': 0.032, 'EVGO': 0.028,
                'TSLA': 0.015, 'RIVN': 0.018, 'F': 0.012
            }
        },

        {
            'company': 'RIVN',
            'type': 'production_delay',
            'sentiment': -0.5,
            'base_outcomes': {
                'RIVN': -0.058, 'LCID': 0.022, 'TSLA': 0.018,
                'F': 0.015, 'LG': -0.012
            }
        },
        {
            'company': 'NIO',
            'type': 'china_expansion',
            'sentiment': 0.75,
            'base_outcomes': {
                'NIO': 0.065, 'CATL': 0.035, 'BYD': -0.012,
                'ALB': 0.025, 'NVDA': 0.015
            }
        },
    ]

    events = []
    start_time = datetime.now() - timedelta(hours=72)

    for i in range(n_events):
        template = events_templates[i % len(events_templates)]

        event_time = start_time + timedelta(hours=i*2.5 + np.random.uniform(-0.5, 0.5))

        outcomes = {}
        for ticker, base_move in template['base_outcomes'].items():
            noise = np.random.normal(0, 0.005)
            outcomes[ticker] = base_move + noise

        event = {
            'id': i,
            'company': template['company'],
            'type': template['type'],
            'sentiment': template['sentiment'] + np.random.normal(0, 0.05),
            'timestamp': event_time.isoformat(),
            'hours_from_start': (event_time - start_time).total_seconds() / 3600,
            'outcomes': outcomes,
            'source_weight': np.random.uniform(0.8, 1.0)
        }

        events.append(event)

    return sorted(events, key=lambda x: x['hours_from_start'])

def create_time_series(events: List[Dict], companies: List[str]) -> Dict[str, np.ndarray]:
    
    n_hours = 72
    time_series = {company: np.zeros(n_hours) for company in companies}

    for event in events:
        event_hour = int(event['hours_from_start'])

        for ticker, price_change in event['outcomes'].items():
            if ticker in time_series and event_hour < n_hours:
                for h in range(event_hour, min(event_hour + 12, n_hours)):
                    decay = np.exp(-0.2 * (h - event_hour))
                    time_series[ticker][h] += price_change * decay

                time_series[ticker] += np.random.normal(0, 0.003, n_hours)

    return time_series

def save_synthetic_data(events: List[Dict], filename: str):
    
    with open(filename, 'w') as f:
        json.dump(events, f, indent=2)

def load_synthetic_data(filename: str) -> List[Dict]:
    
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    events = generate_synthetic_events(30)
    save_synthetic_data(events, '../data/synthetic_events.json')
    print(f"Generated {len(events)} synthetic events")

    for event in events[:5]:
        print(f"\n{event['company']} - {event['type']} (t={event['hours_from_start']:.1f}h)")
        print(f"  Sentiment: {event['sentiment']:.2f}")
        print(f"  Top impacts: {list(event['outcomes'].items())[:3]}")
