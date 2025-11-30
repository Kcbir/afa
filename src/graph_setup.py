import networkx as nx

def build_market_graph():
    
    G = nx.DiGraph()

    companies = {
        'TSLA': {'sector': 'ev_maker', 'size': 100},
        'BYD': {'sector': 'ev_maker', 'size': 80},
        'NIO': {'sector': 'ev_maker', 'size': 40},
        'RIVN': {'sector': 'ev_maker', 'size': 35},
        'LCID': {'sector': 'ev_maker', 'size': 30},
        'F': {'sector': 'legacy_auto', 'size': 70},
        'GM': {'sector': 'legacy_auto', 'size': 65},
        'TM': {'sector': 'legacy_auto', 'size': 90},

        'CATL': {'sector': 'battery', 'size': 75},
        'LG': {'sector': 'battery', 'size': 70},
        'PANW': {'sector': 'battery', 'size': 60},
        'QS': {'sector': 'battery_tech', 'size': 25},
        'ENVX': {'sector': 'battery_tech', 'size': 20},

        'ALB': {'sector': 'lithium', 'size': 50},
        'SQM': {'sector': 'lithium', 'size': 45},
        'LAC': {'sector': 'lithium', 'size': 30},
        'PLL': {'sector': 'lithium', 'size': 25},
        'LTHM': {'sector': 'lithium', 'size': 35},

        'SGL': {'sector': 'graphite', 'size': 20},
        'SYRAH': {'sector': 'graphite', 'size': 15},

        'CHPT': {'sector': 'charging', 'size': 25},
        'BLNK': {'sector': 'charging', 'size': 20},
        'EVGO': {'sector': 'charging', 'size': 18},

        'NVDA': {'sector': 'chips', 'size': 95},
        'AMD': {'sector': 'chips', 'size': 70},
        'INTC': {'sector': 'chips', 'size': 65},
        'MRVL': {'sector': 'chips', 'size': 45},

        'AAPL': {'sector': 'tech', 'size': 100},
        'GOOGL': {'sector': 'tech', 'size': 95},
        'MSFT': {'sector': 'tech', 'size': 98},
    }

    G.add_nodes_from([(name, attrs) for name, attrs in companies.items()])

    relationships = [
        ('TSLA', 'CATL', 0.9, 'supplier'),
        ('TSLA', 'PANW', 0.8, 'supplier'),
        ('TSLA', 'ALB', 0.85, 'indirect_supplier'),
        ('TSLA', 'SQM', 0.75, 'indirect_supplier'),
        ('TSLA', 'LAC', 0.7, 'indirect_supplier'),
        ('TSLA', 'QS', 0.6, 'potential_supplier'),
        ('TSLA', 'NVDA', 0.7, 'tech_partner'),
        ('TSLA', 'CHPT', 0.5, 'competitor_charging'),

        ('BYD', 'CATL', 0.85, 'supplier'),
        ('BYD', 'ALB', 0.8, 'indirect_supplier'),
        ('BYD', 'SQM', 0.75, 'indirect_supplier'),
        ('BYD', 'LTHM', 0.7, 'indirect_supplier'),

        ('NIO', 'CATL', 0.8, 'supplier'),
        ('NIO', 'ALB', 0.7, 'indirect_supplier'),
        ('RIVN', 'LG', 0.75, 'supplier'),
        ('RIVN', 'ALB', 0.7, 'indirect_supplier'),
        ('LCID', 'LG', 0.7, 'supplier'),
        ('F', 'LG', 0.8, 'supplier'),
        ('F', 'CATL', 0.7, 'supplier'),
        ('GM', 'LG', 0.85, 'supplier'),
        ('GM', 'ALB', 0.75, 'indirect_supplier'),
        ('TM', 'PANW', 0.9, 'supplier'),

        ('CATL', 'ALB', 0.9, 'supplier'),
        ('CATL', 'SQM', 0.85, 'supplier'),
        ('CATL', 'PLL', 0.7, 'supplier'),
        ('CATL', 'SGL', 0.8, 'supplier'),
        ('LG', 'ALB', 0.85, 'supplier'),
        ('LG', 'LTHM', 0.8, 'supplier'),
        ('LG', 'SYRAH', 0.7, 'supplier'),
        ('PANW', 'ALB', 0.8, 'supplier'),
        ('PANW', 'SQM', 0.75, 'supplier'),

        ('TSLA', 'BYD', 0.6, 'competitor'),
        ('TSLA', 'NIO', 0.5, 'competitor'),
        ('TSLA', 'RIVN', 0.5, 'competitor'),
        ('NIO', 'BYD', 0.6, 'competitor'),
        ('CHPT', 'BLNK', 0.7, 'competitor'),
        ('CHPT', 'EVGO', 0.7, 'competitor'),
        ('QS', 'ENVX', 0.6, 'competitor'),

        ('AAPL', 'TSLA', 0.4, 'market_correlation'),
        ('GOOGL', 'TSLA', 0.45, 'autonomous_tech'),
        ('NVDA', 'TSLA', 0.7, 'ai_chips'),
        ('NVDA', 'NIO', 0.5, 'ai_chips'),
        ('AMD', 'TSLA', 0.5, 'tech_supplier'),

        ('CHPT', 'TSLA', 0.6, 'infrastructure'),
        ('CHPT', 'RIVN', 0.7, 'infrastructure'),
        ('CHPT', 'F', 0.7, 'infrastructure'),
        ('BLNK', 'TSLA', 0.5, 'infrastructure'),

        ('ALB', 'SQM', 0.8, 'competitor'),
        ('ALB', 'LAC', 0.7, 'competitor'),
        ('SQM', 'LAC', 0.75, 'competitor'),
        ('LAC', 'LTHM', 0.7, 'competitor'),
    ]

    for source, target, weight, rel_type in relationships:
        G.add_edge(source, target, influence=weight, relationship=rel_type)

    return G

def get_company_info(G, ticker):
    
    if ticker in G.nodes():
        return G.nodes[ticker]
    return None

def get_neighbors(G, ticker, relationship_type=None):
    
    if ticker not in G.nodes():
        return []

    neighbors = []
    for neighbor in G.successors(ticker):
        edge_data = G[ticker][neighbor]
        if relationship_type is None or edge_data.get('relationship') == relationship_type:
            neighbors.append({
                'ticker': neighbor,
                'influence': edge_data.get('influence', 0.5),
                'relationship': edge_data.get('relationship', 'unknown')
            })

    return neighbors
