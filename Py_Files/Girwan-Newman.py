import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import networkx as nx


def _find_similarity(X, measure="euclidean", threshold=0.7):
    """
    Finds similarity between samples in X using the specified distance measure and threshold.
    """
    distances = squareform(pdist(X, metric=measure))
    similarities = 1 / (1 + distances)
    G = nx.Graph()
    n_samples = X.shape[0]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if similarities[i, j] > threshold:
                G.add_edge(i, j, weight=similarities[i, j])

    return G


def assign_communities(G):
    """
    Assigns community IDs to nodes in a graph based on connected components.
    """
    communities = list(nx.connected_components(G))
    community_map = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_map[node] = community_id
    return community_map


def plot_graph_with_communities_from_df(G, df, node_col, community_col="community"):
    """
    Plots a graph with nodes colored based on their communities from a DataFrame.
    """
    communities = df[community_col].unique()
    color_map = {community: plt.cm.rainbow(
        i / len(communities)) for i, community in enumerate(communities)}
    node_colors = df.set_index(node_col)[community_col].map(color_map)
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=True,
        node_color=[node_colors[node] for node in G.nodes],
        node_size=700,
        edge_color="gray",
        font_weight="bold"
    )
    if len(G.nodes) > 1000:
        edge_labels = {} 
    else:
        edge_labels = {(u, v): f"{u}-{v}" for u, v in G.edges}

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels
    )


if __name__ == "__main__":
    from preproccess_data import preprocess_data

    column_names = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num',
                    'Marital-status', 'Occupation', 'Relationship', 'Race',
                    'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
                    'Native-country', 'Income']
    df = pd.read_csv("Data/adult.csv", names=column_names)

    test_df = df.iloc[:25000]
    X_transformed, names = preprocess_data(test_df)
    graph = _find_similarity(
        X_transformed, measure="euclidean", threshold=0.85)
    community_map = assign_communities(graph)
    test_df = test_df.reset_index()
    test_df["community"] = test_df["index"].map(community_map)

    plot_graph_with_communities_from_df(
        graph, test_df, node_col="index", community_col="community")

    print("sdsd")
