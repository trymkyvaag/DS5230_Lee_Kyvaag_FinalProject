import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def load_filtered_rules(file_path, nrows=None):
    """
    Load the filtered rules from the provided Excel file.
    """
    return pd.read_excel(file_path, nrows=nrows)


def group_antecedents(rules):
    """
    Group antecedents and store the corresponding consequents and metrics.
    """
    grouped_rules = {}
    for _, row in tqdm(rules.iterrows(), total=len(rules), desc="Grouping Antecedents"):
        antecedents = frozenset(eval(row['antecedents_features']))
        consequents = frozenset(eval(row['consequents_features']))
        confidence = row['confidence']
        lift = row['lift']

        if antecedents not in grouped_rules:
            grouped_rules[antecedents] = {'consequents': {}}
        if consequents not in grouped_rules[antecedents]['consequents']:
            grouped_rules[antecedents]['consequents'][consequents] = {
                'confidence': confidence, 'lift': lift}
    return grouped_rules


# def cluster_antecedents(grouped_rules, max_group_size=100):
#     """
#     Cluster antecedents into groups to reduce the number of nodes in the graph.
#     Groups antecedents with similar features together.
#     """
#     clustered_rules = defaultdict(lambda: {'consequents': {}})
#     antecedent_map = {}

#     for antecedents, details in grouped_rules.items():
#         antecedent_features = frozenset(antecedents)
#         # Use a fixed number of features to group
#         key = tuple(sorted(antecedent_features)[:max_group_size])

#         # Merge into the cluster
#         for consequents, metrics in details['consequents'].items():
#             clustered_rules[key]['consequents'][consequents] = metrics

#         # Map original antecedents to the cluster for visualization purposes
#         antecedent_map[antecedents] = key

#     return dict(clustered_rules), antecedent_map


# def filter_high_income_rules(grouped_rules):
#     """
#     Filter the rules related to high income.
#     """
#     high_income_rules = {}
#     for antecedents, details in grouped_rules.items():
#         # Check if 'high_income' is in the antecedents and its value is True or False
#         high_income_consequents = {
#             k: v
#             for k, v in details['consequents'].items()
#             if any('high_income' in item and ('True' in item or 'False' in item) for item in antecedents)
#         }
#         if high_income_consequents:
#             high_income_rules[antecedents] = {
#                 'consequents': high_income_consequents}
#     return high_income_rules


# def create_simple_graph(grouped_rules):
#     """
#     Create a simple directed graph based on the grouped rules.
#     """
#     G = nx.DiGraph()
#     for antecedents, details in grouped_rules.items():
#         antecedents_label = ', '.join(sorted(antecedents))
#         for consequents, metrics in details['consequents'].items():
#             consequents_label = ', '.join(sorted(consequents))
#             confidence = metrics['confidence']
#             lift = metrics['lift']
#             G.add_edge(antecedents_label, consequents_label,
#                        weight=lift, confidence=confidence)
#     return G


# def visualize_graph(G):

#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(G, k=0.3, iterations=20)
#     nx.draw_networkx_nodes(G, pos, node_size=700,
#                            node_color='lightblue', alpha=0.7)
#     nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, font_size=10,
#                             font_color='black', font_weight='bold')
#     plt.title("High Income Rule Relationships")
#     plt.axis("off")
#     plt.show()


def filter_high_income_consequents(grouped_rules):
    high_income_rules = {'True': {}, 'False': {}}

    for antecedents, details in grouped_rules.items():
        for consequents, metrics in details['consequents'].items():
            if 'high_income: True' in consequents:
                high_income_rules['True'][antecedents] = {
                    'lift': metrics['lift'], 'confidence': metrics['confidence']}
            elif 'high_income: False' in consequents:
                high_income_rules['False'][antecedents] = {
                    'lift': metrics['lift'], 'confidence': metrics['confidence']}

    return high_income_rules

# Cause of similar outputs?


def rank_and_limit_antecedents(high_income_rules, top_n=5):
    ranked_rules = {'True': [], 'False': []}

    for outcome, rules in high_income_rules.items():
        sorted_rules = sorted(
            rules.items(), key=lambda x: x[1]['lift'], reverse=False)
        ranked_rules[outcome] = sorted_rules[:top_n]

    return ranked_rules


def create_filtered_graph(ranked_rules):
    """
    Create a simplified directed graph for high income rules.
    """
    G = nx.DiGraph()
    # Add nodes and edges for high_income: True and high_income: False
    for outcome, rules in ranked_rules.items():
        target_node = f"high_income: {outcome}"
        for antecedents, metrics in rules:
            antecedents_label = ', '.join(sorted(antecedents))
            confidence = metrics['confidence']
            lift = metrics['lift']
            G.add_edge(antecedents_label, target_node,
                       weight=lift, confidence=confidence)

    return G

# I want to find better wayy to write edgbes


def truncate_label(label, max_length=50):
    """
    Truncate a label to ensure it fits within the graph visualization.
    """
    if len(label) > max_length:
        return label[:max_length] + "..."
    return label


def visualize_graph_with_lift(G):
    """
    Visualize the directed graph with lift values on the edges and truncated node labels.
    """
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.3, iterations=20)
    truncated_labels = {node: truncate_label(node) for node in G.nodes}
    nx.draw_networkx_nodes(G, pos, node_size=700,
                           node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')

    nx.draw_networkx_labels(G, pos, labels=truncated_labels,
                            font_size=10, font_color='black', font_weight='bold')

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {key: f"Lift: {value:.2f}" for key,
                   value in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    # add var to f string for high/low
    plt.title("High Income Rule Relationships with Lift Values")
    plt.axis("off")
    plt.show()


def export_labels_to_csv(G, file_name="graph_labels.csv"):
    """
    Export a mapping of truncated labels to full labels to a CSV file.
    """
    full_labels = {node: node for node in G.nodes}
    truncated_labels = {node: truncate_label(node) for node in G.nodes}

    data = [
        {"Truncated Label": truncated_labels[node],
            "Full Label": full_labels[node]}
        for node in G.nodes
    ]
    pd.DataFrame(data).to_csv(file_name, index=False)
    print(f"Labels exported to {file_name}")


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets: 

    #### intersect / union ####

    """

    if isinstance(set1, str):
        try:
            set1 = frozenset(eval(set1))
        except:
            pass

    if isinstance(set2, str):
        try:
            set2 = frozenset(eval(set2))
        except:

            pass

    if not isinstance(set1, frozenset):
        set1 = frozenset([set1])
    if not isinstance(set2, frozenset):
        set2 = frozenset([set2])

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def filter_similar_rules(grouped_rules, threshold=0.8):
    """
    Filter out similar rules based on Jaccard similarity.
    Only keep one rule from each similar grou
    """
    antecedents_list = list(grouped_rules.keys())
    filtered_indices = set()

    for i, antecedents_i in enumerate(antecedents_list):
        if i in filtered_indices:
            continue  # Skip already-selected rules
        filtered_indices.add(i)

        for j, antecedents_j in enumerate(antecedents_list):
            if j <= i or j in filtered_indices:
                continue  # Skip previously checked or selected rules

            similarity = jaccard_similarity(antecedents_i, antecedents_j)
            if similarity >= threshold:
                # If too similar, exclude the rule `j`
                continue

    filtered_rules = {
        antecedents_list[i]: grouped_rules[antecedents_list[i]] for i in filtered_indices
    }
    return filtered_rules


def main():
    file_path = 'filtered_rules.xlsx'

    rules = load_filtered_rules(file_path, nrows=100)
    print("Grouping antecedents...")
    grouped_rules = group_antecedents(rules)
    print("Filtering high income consequents...")
    high_income_rules = filter_high_income_consequents(grouped_rules)
    print("Filtering similar rules...")
    filtered_high_income_rules = filter_similar_rules(
        high_income_rules, threshold=0.15)
    print("Ranking and limiting antecedents...")
    ranked_rules = rank_and_limit_antecedents(
        filtered_high_income_rules, top_n=5)
    print("Creating filtered graph for high income rules...")
    G = create_filtered_graph(ranked_rules)
    visualize_graph_with_lift(G)
    export_labels_to_csv(G, "high_income_graph_labels.csv")


if __name__ == "__main__":
    main()
