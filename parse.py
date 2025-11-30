import json
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def load_level(path):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------------------
# Determine arrow direction from first 2 nodes
# -----------------------------------------

def get_direction(nodes):
    x1, y1 = nodes[0]["x"], nodes[0]["y"]
    x2, y2 = nodes[1]["x"], nodes[1]["y"]

    if x2 == x1 and y2 < y1:
        return "UP"
    if x2 == x1 and y2 > y1:
        return "DOWN"
    if y2 == y1 and x2 > x1:
        return "RIGHT"
    if y2 == y1 and x2 < x1:
        return "LEFT"
    return None

# -----------------------------------------
# Check if arrow A is blocked by arrow B
# -----------------------------------------

def is_blocked(A, B):
    head = A["nodes"][0]
    hx, hy = head["x"], head["y"]
    direction = get_direction(A["nodes"])

    for n in B["nodes"]:
        bx, by = n["x"], n["y"]

        if direction == "UP":
            if bx == hx and by > hy:     # any node above A
                return True

        elif direction == "DOWN":
            if bx == hx and by < hy:
                return True

        elif direction == "LEFT":
            if by == hy and bx > hx:
                return True

        elif direction == "RIGHT":
            if by == hy and bx < hx:
                return True

    return False

# -----------------------------------------
# Build dependency graph
# -----------------------------------------

def build_dependency_graph(level):
    arrows = level["arrows"]
    graph = defaultdict(list)

    for i, A in enumerate(arrows):
        for j, B in enumerate(arrows):
            if i == j:
                continue
            if is_blocked(A, B):
                graph[i].append(j)   # arrow i depends on j

    return graph

# -----------------------------------------
# Print result
# -----------------------------------------

def print_graph(graph):
    if not graph:
        print("No dependencies found")
        return
    for arrow, deps in graph.items():
        print(f"Arrow {arrow} depends on {deps}")


def draw_dependency_graph(graph, arrows):
    G = nx.DiGraph()

    # Add nodes (arrow ids)
    for i in range(len(arrows)):
        color = arrows[i]["color"]
        G.add_node(i, label=f"{i}\ncolor={color}")

    # Add edges
    for a, deps in graph.items():
        for b in deps:
            G.add_edge(a, b)

    # Layout for nice drawing
    pos = nx.spring_layout(G, seed=42, k=0.7)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#87CEEB")

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)

    # Labels
    labels = {i: f"A{i}\nC{arrows[i]['color']}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("ArrowPuzzle Dependency Graph")
    plt.axis("off")
    plt.show()

def draw_dependency_graph_interactive(graph, arrows, output_html="dependency_graph.html"):
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut()  # force simulation layout

    # Add nodes
    for i in range(len(arrows)):
        arrow = arrows[i]
        color = arrow.get("color", "gray")

        net.add_node(
            i,
            label=f"A{i}",
            title=f"Arrow {i}<br>Color: {color}<br>Dir: {get_direction(arrow['nodes'])}",
            color=color
        )

    # Add edges (dependencies)
    for a, deps in graph.items():
        for b in deps:
            net.add_edge(a, b, title=f"A{a} depends on A{b}")

    # Generate HTML file
    net.write_html(output_html)
    print(f"Interactive dependency graph generated: {output_html}")

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    level = load_level("/Users/hoangnguyen/Documents/py/Arrow/asset-game-level/lv8.json")
    graph = build_dependency_graph(level)
    # draw_dependency_graph(graph, level["arrows"])
    # draw_dependency_graph_interactive(graph, level["arrows"])
    print_graph(graph)
