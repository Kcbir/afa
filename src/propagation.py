import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

def propagate_attention(G: nx.DiGraph,
                       source: str,
                       initial_intensity: float,
                       lambda_decay: float = 0.3,
                       beta: float = 1.5,
                       max_distance: int = 4) -> Dict[str, float]:
    
    scores = {}

    for target in G.nodes():
        if target == source:
            scores[target] = initial_intensity
            continue

        try:
            distance = nx.shortest_path_length(G, source, target)

            if distance > max_distance:
                scores[target] = 0.0
            else:
                spatial_decay = (1 - distance / max_distance) ** beta

                path = nx.shortest_path(G, source, target)
                path_influence = 1.0
                for i in range(len(path) - 1):
                    edge_data = G[path[i]][path[i+1]]
                    path_influence *= edge_data.get('influence', 0.5)

                scores[target] = initial_intensity * spatial_decay * path_influence

        except nx.NetworkXNoPath:
            scores[target] = 0.0

    return scores

def apply_temporal_decay(scores: Dict[str, float],
                        time_elapsed: float,
                        lambda_decay: float = 0.3) -> Dict[str, float]:
    
    decay_factor = np.exp(-lambda_decay * time_elapsed)
    return {k: v * decay_factor for k, v in scores.items()}

def propagate_with_time(G: nx.DiGraph,
                       source: str,
                       initial_intensity: float,
                       time_hours: float,
                       lambda_decay: float = 0.3,
                       beta: float = 1.5) -> Dict[str, float]:
    
    spatial_scores = propagate_attention(G, source, initial_intensity, lambda_decay, beta)
    return apply_temporal_decay(spatial_scores, time_hours, lambda_decay)

def build_transition_matrix(G: nx.DiGraph) -> Tuple[np.ndarray, List[str]]:
    
    nodes = list(G.nodes())
    n = len(nodes)
    P = np.zeros((n, n))

    for i, node_i in enumerate(nodes):
        neighbors = list(G.successors(node_i))

        if not neighbors:
            P[i][i] = 1.0
            continue

        total_weight = sum(G[node_i][neighbor].get('influence', 0.5)
                          for neighbor in neighbors)

        if total_weight == 0:
            total_weight = len(neighbors)

        for j, node_j in enumerate(nodes):
            if node_j in neighbors:
                influence = G[node_i][node_j].get('influence', 0.5)
                P[i][j] = influence / total_weight

    return P, nodes

def simulate_markov_propagation(G: nx.DiGraph,
                                source: str,
                                initial_intensity: float,
                                n_steps: int = 10) -> Dict[int, Dict[str, float]]:
    
    P, nodes = build_transition_matrix(G)

    state = np.zeros(len(nodes))
    source_idx = nodes.index(source)
    state[source_idx] = initial_intensity

    history = {0: dict(zip(nodes, state))}

    for step in range(1, n_steps + 1):
        state = P.T @ state
        state *= 0.95
        history[step] = dict(zip(nodes, state))

    return history

def calculate_attention_metrics(scores: Dict[str, float]) -> Dict:
    
    values = np.array(list(scores.values()))

    return {
        'total_attention': np.sum(values),
        'mean_attention': np.mean(values),
        'max_attention': np.max(values),
        'std_attention': np.std(values),
        'entropy': -np.sum(values * np.log(values + 1e-10)) if np.sum(values) > 0 else 0,
        'top_5': sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    }

def multi_source_propagation(G: nx.DiGraph,
                             events: List[Tuple[str, float, float]],
                             current_time: float,
                             lambda_decay: float = 0.3) -> Dict[str, float]:
    
    aggregated_scores = {node: 0.0 for node in G.nodes()}

    for source, intensity, event_time in events:
        time_elapsed = current_time - event_time

        if time_elapsed < 0:
            continue

        scores = propagate_with_time(G, source, intensity, time_elapsed, lambda_decay)

        for node, score in scores.items():
            aggregated_scores[node] += score

    return aggregated_scores

def find_critical_nodes(G: nx.DiGraph, top_n: int = 10) -> List[Tuple[str, float]]:
    
    influence_scores = {}

    for node in G.nodes():
        scores = propagate_attention(G, node, 1.0)
        total_influence = sum(scores.values())
        influence_scores[node] = total_influence

    sorted_nodes = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_n]
